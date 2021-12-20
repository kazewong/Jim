import numpy as np
import bilby
import jax
import jax.numpy as jnp


from jax.config import config

from jaxgw.sampler.NF_proposal import nf_metropolis_kernel, nf_metropolis_sampler
config.update("jax_enable_x64", True)

from jaxgw.gw.likelihood.detector_projection import construct_arm, detector_tensor, antenna_response, get_detector_response

from jaxgw.gw.likelihood.utils import inner_product
from jaxgw.gw.likelihood.detector_preset import get_H1, get_L1
from jaxgw.gw.waveform.TaylorF2 import TaylorF2
from jaxgw.gw.waveform.IMRPhenomC import IMRPhenomC
from jax import random, grad, jit, vmap, jacfwd, jacrev, value_and_grad, pmap

from jaxgw.sampler.Gaussian_random_walk import rw_metropolis_sampler
from jaxgw.sampler.maf import MaskedAutoregressiveFlow
from jaxgw.sampler.realNVP import RealNVP
from jax.scipy.stats import multivariate_normal
from flax.training import train_state  # Useful dataclass to keep train state
import optax                           # Optimizers


true_m1 = 30.
true_m2 = 20.
true_ld = 300.
true_phase = 0.
true_gt = 0.

injection_parameters = dict(
	mass_1=true_m1, mass_2=true_m2, spin_1=0.0, spin_2=0.0, luminosity_distance=true_ld, theta_jn=0.4, psi=2.659,
	phase_c=true_phase, t_c=true_gt, ra=1.375, dec=-1.2108)


#guess_parameters = dict(m1=true_m1, m2=true_m2)

guess_parameters = dict(
	mass_1=true_m1*0.99, mass_2=true_m2*1.01, luminosity_distance=true_ld, theta_jn=0.4, psi=2.659,
	phase_c=true_phase, t_c=true_gt, ra=1.375, dec=-1.2108)



# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(['H1'])
ifos.set_strain_data_from_power_spectral_densities(
	sampling_frequency=2048, duration=1,
	start_time=- 3)

psd = ifos[0].power_spectral_density_array
psd_frequency = ifos[0].frequency_array

psd_frequency = psd_frequency[jnp.isfinite(psd)]
psd = psd[jnp.isfinite(psd)]

waveform = IMRPhenomC(psd_frequency, injection_parameters)
#waveform = TaylorF2(psd_frequency, injection_parameters)
H1, H1_vertex = get_H1()
L1, L1_vertex = get_L1()
strain_H1 = get_detector_response(psd_frequency, waveform, injection_parameters, H1, H1_vertex)
strain_L1 = get_detector_response(psd_frequency, waveform, injection_parameters, L1, L1_vertex)

print('SNR of the event in H1: '+str(np.sqrt(inner_product(strain_H1,strain_H1,psd_frequency,psd))))
print('SNR of the event in L1: '+str(np.sqrt(inner_product(strain_L1,strain_L1,psd_frequency,psd))))

@jit
def single_detector_likelihood(params, data, data_f, PSD, detector, detector_vertex):
	waveform = IMRPhenomC(data_f, params)
#	waveform = TaylorF2(data_f, params)
	waveform = get_detector_response(data_f, waveform, params, detector, detector_vertex)
	match_filter_SNR = inner_product(waveform, data, data_f, PSD)
	optimal_SNR = inner_product(waveform, waveform, data_f, PSD)
	return -(-2*match_filter_SNR + optimal_SNR)/2#, match_filter_SNR, optimal_SNR

#@jit
#def logprob_wrap(m1, m2):
#	params = dict(mass_1=m1, mass_2=m2, spin_1=0, spin_2=0, luminosity_distance=true_ld, phase_c=true_phase, t_c=true_gt, theta_jn=0.4, psi=2.659, ra=1.375, dec=-1.2108)
#	return single_detector_likelihood(params, strain_H1, psd_frequency, psd, H1, H1_vertex)+single_detector_likelihood(params, strain_L1, psd_frequency, psd, L1, L1_vertex)
#
@jit
def logprob_wrap(mass_1, mass_2, luminosity_distance, phase_c, t_c, theta_jn, psi, ra, dec):
	params = dict(mass_1=mass_1, mass_2=mass_2, spin_1=0, spin_2=0, luminosity_distance=true_ld, phase_c=phase_c, t_c=t_c, theta_jn=theta_jn, psi=psi, ra=ra, dec=dec)
#	params = dict(mass_1=mass_1, mass_2=mass_2, spin_1=0, spin_2=0, luminosity_distance=true_ld, theta_jn=0.4, psi=2.659, phase_c=true_phase, t_c=true_gt, ra=1.375, dec=-1.2108)
	return single_detector_likelihood(params, strain_H1, psd_frequency, psd, H1, H1_vertex)+single_detector_likelihood(params, strain_L1, psd_frequency, psd, L1, L1_vertex)

likelihood = lambda x: logprob_wrap(*x)
para_logp = jit(jax.vmap(likelihood))

#### Sampling ####

def train_step(model, state, batch):
	def loss(params):
		y, log_det = model.apply({'params': params},batch)
		mean = jnp.zeros((batch.shape[0],model.n_features))
		cov = jnp.repeat(jnp.eye(model.n_features)[None,:],batch.shape[0],axis=0)
		log_det = log_det + multivariate_normal.logpdf(y,mean,cov)
		return -jnp.mean(log_det)
	grad_fn = jax.value_and_grad(loss)
	value, grad = grad_fn(state.params)
	state = state.apply_gradients(grads=grad)
	return value,state

train_step = jax.jit(train_step,static_argnums=(0,))

def train_flow(rng, model, state, data):

    def train_epoch(state, train_ds, batch_size, epoch, rng):
        """Train for a single epoch."""
        train_ds_size = len(train_ds)
        steps_per_epoch = train_ds_size // batch_size

        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        for perm in perms:
            batch = train_ds[perm, ...]
            value, state = train_step(model, state, batch)

        return value, state

    for epoch in range(1, num_epochs + 1):
        print('Epoch %d' % epoch)
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        value, state = train_epoch(state, data, batch_size, epoch, input_rng)
        print('Train loss: %.3f' % value)

    return rng, state

def sample_nf(model, param, rng_key,n_sample):
    rng_key, subkey = random.split(rng_key)
    samples = model.apply({'params': param}, subkey, n_sample,param, method=model.sample)
    samples = jnp.flip(samples[0],axis=1)
    return rng_key,samples

n_dim = 9
n_samples = 100000
nf_samples = 100
n_chains = 20
learning_rate = 0.01
momentum = 0.9
num_epochs = 300
batch_size = 10000
precompiled = False

print("Preparing RNG keys")
rng_key = jax.random.PRNGKey(42)
rng_key_mcmc, rng_key_nf = jax.random.split(rng_key,2)

rng_keys_mcmc = jax.random.split(rng_key_mcmc, n_chains)  # (nchains,)
rng_keys_nf, init_rng_keys_nf = jax.random.split(rng_key_nf,2)

print("Initializing MCMC model and normalizing flow model.")

initial_position = (jnp.zeros((9, n_chains)).T + jnp.array(list(guess_parameters.values()))).T #(n_dim, n_chains)

#model = MaskedAutoregressiveFlow(n_dim,64,4)
model = RealNVP(10,n_dim,64, 1)
params = model.init(init_rng_keys_nf, jnp.ones((1,n_dim)))['params']

run_mcmc = jax.vmap(rw_metropolis_sampler, in_axes=(0, None, None, 1),
                    out_axes=0)

tx = optax.adam(learning_rate, momentum)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def sampling_loop(rng_keys_nf, rng_keys_mcmc, model, state, initial_position):
	rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples, likelihood, initial_position)
	flat_chain = positions.reshape(-1,n_dim)	
	rng_keys_nf, nf_chain, log_prob, log_prob_nf = nf_metropolis_sampler(rng_keys_nf, nf_samples, model, state.params , para_logp, positions[:,-1])

	positions = jnp.concatenate((positions,nf_chain),axis=1)
	return rng_keys_nf, rng_keys_mcmc, state, positions

last_step = initial_position
chains = []
for i in range(15):
	# rng_keys_nf, rng_keys_mcmc, state, positions = sampling_loop(rng_keys_nf, rng_keys_mcmc, model, state, last_step)
	# last_step = positions[:,-1].T
	rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples, likelihood, initial_position)
	last_step = last_step.T
	# if i%5 == 0:
		# rng_keys_nf, state = train_flow(rng_key_nf, model, state, positions.reshape(-1,n_dim))
	chains.append(positions)

chains = np.concatenate(chains,axis=1)
nf_samples = sample_nf(model, state.params, rng_keys_nf, 10000)
import time
from jimgw.jim import Jim
from jimgw.detector import H1, L1
from jimgw.likelihood import TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import corner
import numpy as np

jax.config.update('jax_enable_x64', True)

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
start = gps - 2
end = gps + 2
fmin = 20.
fmax = 1024.

ifos = ['H1', 'L1']

H1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

likelihood = TransientLikelihoodFD([H1, L1], RippleIMRPhenomD(), gps, 4, 2)
prior = Uniform(
    xmin = [10, 0.125, -1., -1., 0., -0.05, 0., -1, 0., 0.,-1.],
    xmax = [80., 1., 1., 1., 2000., 0.05, 2*jnp.pi, 1., jnp.pi, 2*jnp.pi, 1.],
    naming = ["M_c", "q", "s1_z", "s2_z", "d_L", "t_c", "phase_c", "cos_iota", "psi", "ra", "sin_dec"],
    transforms = {"q": ("eta", lambda q: q/(1+q)**2),
                 "cos_iota": ("iota",lambda cos_iota: jnp.arccos(jnp.arcsin(jnp.sin(cos_iota/2*jnp.pi))*2/jnp.pi)),
                 "sin_dec": ("dec",lambda sin_dec: jnp.arcsin(jnp.arcsin(jnp.sin(sin_dec/2*jnp.pi))*2/jnp.pi))} # sin and arcsin are periodize cos_iota and sin_dec
)


mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[1,1].set(1e-3)
mass_matrix = mass_matrix.at[5,5].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix*3e-3}

jim = Jim(likelihood, 
        prior,
        n_loop_training=10,
        n_loop_production = 10,
        n_local_steps=300,
        n_global_steps=300,
        n_chains=500,
        n_epochs=300,
        learning_rate = 0.001,
        momentum = 0.9,
        batch_size = 50000,
        use_global=True,
        keep_quantile=0.,
        train_thinning = 40,
        local_sampler_arg = local_sampler_arg,
        )

if True:
        key, subkey = jax.random.split(jax.random.PRNGKey(1234))
        jim.sample(subkey)
        chains = jim.get_samples(training=True)
        chains = np.array(chains)
        figure = corner.corner(
        chains.reshape(-1, 11), labels = prior.naming, max_n_ticks = 4, plot_datapoints=False, quantiles=(0.16, 0.5, 0.84), levels=(0.68,), show_titles = True)
        figure.set_size_inches(16, 16)
        figure.suptitle("Visualize PE run", fontsize = 33)
        plt.show(block=False)
        plt.savefig("/mnt/home/averhaeghe/jim/example/First_run_GW150914.png")
jim.maximize_likelihood([prior.xmin, prior.xmax])
jim.sample(jax.random.PRNGKey(42))

# Import packages
import numpy as np
import jax.numpy as jnp
import jax
from lal import GreenwichMeanSiderealTime
from jax.config import config

config.update("jax_enable_x64", True)

import os
import sys
from ripple import ms_to_Mc_eta
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc, gen_IMRPhenomD
from jaxgw.PE.detector_preset import *
from jaxgw.PE.single_event_likelihood import single_detector_likelihood
from jaxgw.PE.detector_projection import make_detector_response, get_detector_response
from jaxgw.PE.generate_noise import generate_noise
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from jaxgw.PE.utils import inner_product
from jax import grad, vmap
from functools import partial

import argparse
import yaml

from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
sys.path.append("/Users/thomasedwards/Dropbox/Work/Gravitational_Waves/JaxGW")

parser = argparse.ArgumentParser(description="Injection test")

parser.add_argument("--config", type=str, default="config.yaml", help="config file")

# Add noise parameters to parser
parser.add_argument(
    "--seed", type=int, default=None, help="seed for random number generator"
)
parser.add_argument("--f_sampling", type=int, default=None, help="sampling frequency")
parser.add_argument("--duration", type=int, default=None, help="duration of the data")
parser.add_argument("--fmin", type=float, default=None, help="minimum frequency")
parser.add_argument("--ifos", nargs="+", default=None, help="list of detectors")

# Add injection parameters to parser
parser.add_argument(
    "--m1", type=float, default=None, help="mass of the first component"
)
parser.add_argument(
    "--m2", type=float, default=None, help="mass of the second component"
)
parser.add_argument(
    "--chi1", type=float, default=None, help="dimensionless spin of the first component"
)
parser.add_argument(
    "--chi2",
    type=float,
    default=None,
    help="dimensionless spin of the second component",
)
parser.add_argument(
    "--dist_mpc", type=float, default=None, help="distance in megaparsecs"
)
parser.add_argument("--tc", type=float, default=None, help="coalescence time")
parser.add_argument("--phic", type=float, default=None, help="phase of coalescence")
parser.add_argument("--inclination", type=float, default=None, help="inclination angle")
parser.add_argument(
    "--polarization_angle", type=float, default=None, help="polarization angle"
)
parser.add_argument("--ra", type=float, default=None, help="right ascension")
parser.add_argument("--dec", type=float, default=None, help="declination")
parser.add_argument(
    "--heterodyne_bins",
    type=int,
    default=101,
    help="number of bins for heterodyne likelihood",
)

# Add sampler parameters to parser

parser.add_argument("--n_dim", type=int, default=None, help="number of parameters")
parser.add_argument("--n_chains", type=int, default=None, help="number of chains")
parser.add_argument(
    "--n_loop_training", type=int, default=None, help="number of training loops"
)
parser.add_argument(
    "--n_loop_production", type=int, default=None, help="number of production loops"
)
parser.add_argument(
    "--n_local_steps", type=int, default=None, help="number of local steps"
)
parser.add_argument(
    "--n_global_steps", type=int, default=None, help="number of global steps"
)
parser.add_argument("--learning_rate", type=float, default=None, help="learning rate")
parser.add_argument(
    "--max_samples", type=int, default=None, help="maximum number of samples"
)
parser.add_argument(
    "--momentum", type=float, default=None, help="momentum during training"
)
parser.add_argument("--num_epochs", type=int, default=None, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=None, help="batch size")
parser.add_argument(
    "--stepsize", type=float, default=None, help="stepsize for Local sampler"
)

# Add output parameters to parser

parser.add_argument("--output_path", type=str, default=None, help="output file path")

# parser

args = parser.parse_args()
opt = vars(args)
args = yaml.load(open(opt["config"], "r"), Loader=yaml.FullLoader)
opt.update(args)
args = opt

# Fetch noise parameters

print("Constructing detectors")
print("Making noises")

seed = args["seed"]
f_sampling = args["f_sampling"]
duration = args["duration"]
fmin = args["fmin"]
ifos = args["ifos"]

freqs, psd_dict, noise_dict = generate_noise(
    seed + 1234, f_sampling, duration, fmin, ifos
)

# Fetch injection parameters and inject signal

m1 = args["m1"]
m2 = args["m2"]
chi1 = args["chi1"]
chi2 = args["chi2"]
dist_mpc = args["dist_mpc"]
tc = args["tc"]
phic = args["phic"]
inclination = args["inclination"]
polarization_angle = args["polarization_angle"]
ra = args["ra"]
dec = args["dec"]

Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))

H1 = get_H1()
H1_response = make_detector_response(H1[0], H1[1])
L1 = get_L1()
L1_response = make_detector_response(L1[0], L1[1])

f_ref = fmin
trigger_time = 1126259462.4
post_trigger_duration = 2
epoch = duration - post_trigger_duration
gmst = GreenwichMeanSiderealTime(trigger_time)

true_param = jnp.array(
    [Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle, ra, dec]
)

f_list = freqs[freqs > fmin]
psd_list = [psd_dict["H1"], psd_dict["L1"]]
waveform_generator = lambda f_, theta_: gen_IMRPhenomD_hphc(f_, theta_, f_ref)
hp_real = lambda theta, f: waveform_generator(f, theta)[0].real
hp_imag = lambda theta, f: waveform_generator(f, theta)[0].imag


# def make_simple_logL(waveform_model, f, PSD):
#     def logL_(theta):
#         extrinsic = jnp.array([chi1, chi2, dist_mpc, tc, phic, inclination])
#         params = jnp.append(theta, extrinsic)
#         hp, hc = waveform_model(f, params)
#         index = jnp.where((jnp.abs(hp)) > 0)
#         logL = inner_product(hp[index], hp[index], f[index], PSD[index])
#         return -logL / 2.0

#     return logL_


PSD_vals = jax.numpy.interp(f_list, freqs, psd_list[0])
# plt.figure(figsize=(7, 5))
# plt.semilogy(f_list, PSD_vals)
# plt.savefig("plots/PSDtest.pdf", bbox_inches="tight")

# def make_logL(waveform_model, f, PSD, detector, gmst, epoch):
#     logL_ = lambda theta: single_detector_likelihood(
#         waveform_model, theta, f, PSD, detector, gmst, epoch
#     )
#     return logL_
# logL = make_logL(waveform_generator, f_list, psd_list[0], H1, gmst, epoch)

# logL = make_simple_logL(waveform_generator, f_list, PSD_vals)


strain_real = lambda params, f: get_detector_response(
    waveform_generator, params, f, H1, gmst, epoch
).real
strain_imag = lambda params, f: get_detector_response(
    waveform_generator, params, f, H1, gmst, epoch
).imag


def get_strain_derivatives(theta):
    hp_grad_real = vmap(partial(grad(strain_real), theta))(f_list)
    hp_grad_imag = vmap(partial(grad(strain_imag), theta))(f_list)
    return hp_grad_real + hp_grad_imag * 1j


strain_grad = get_strain_derivatives(true_param)


# Lets go a bit further and try to take more general derivatives
# def get_waveform_derivatives(theta):
#     extrinsic = jnp.array([chi1, chi2, dist_mpc, tc, phic, inclination])
#     params = jnp.append(theta, extrinsic)
#     hp_grad_real = vmap(partial(grad(hp_real), params))(f_list)
#     hp_grad_imag = vmap(partial(grad(hp_imag), params))(f_list)
#     return hp_grad_real + hp_grad_imag * 1j


# param_subset = jnp.array([Mc, eta])
# hp_grad = get_waveform_derivatives(param_subset)

F = np.zeros([2, 2])
for i in range(2):
    for j in range(2):
        F[i, j] = inner_product(strain_grad[:, i], strain_grad[:, j], f_list, PSD_vals)


def plot_contours(fisher, pos, nstd=1.0, ax=None, **kwargs):
    """
    Plot 2D parameter contours given a Hessian matrix of the likelihood
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    mat = fisher
    cov = np.linalg.inv(mat)
    sigma_marg = lambda i: np.sqrt(cov[i, i])

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    s1 = 1.5 * nstd * sigma_marg(0)
    s2 = 1.5 * nstd * sigma_marg(1)
    ax.set_xlim(pos[0] - s1, pos[0] + s1)
    ax.set_ylim(pos[1] - s2, pos[1] + s2)
    plt.draw()
    return ellip


print("Fisher")

print("Fisher", F)
cov = np.linalg.inv(F)
print("Sqrt Inverse Fisher", np.sqrt(cov))

plot_contours(F, true_param[:2], fill=False)
plt.xlabel("Mchirp")
plt.ylabel("eta")
plt.savefig("plots/test.pdf", bbox_inches="tight")

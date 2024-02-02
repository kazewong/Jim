# Import packages
import numpy as np
import jax.numpy as jnp
import jax
from lal import GreenwichMeanSiderealTime

import os
import sys
from ripple import ms_to_Mc_eta
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar, gen_IMRPhenomD
from jaxgw.PE.detector_preset import *
from jaxgw.PE.single_event_likelihood import single_detector_likelihood
from jaxgw.PE.detector_projection import make_detector_response, get_detector_response
from jaxgw.PE.generate_noise import generate_noise
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from jaxgw.PE.utils import inner_product
from jax import grad, vmap
from functools import partial
import time
from jax.config import config

config.update("jax_enable_x64", True)

import argparse
import yaml

from tqdm import tqdm
from functools import partialmethod


# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
sys.path.append("/Users/thomasedwards/Dropbox/Work/Gravitational_Waves/JaxGW")

parser = argparse.ArgumentParser(description="Injection test")

parser.add_argument("--config", type=str, default="config.yaml", help="config file")

# Add noise parameters to parser
parser.add_argument("--f_sampling", type=int, default=None, help="sampling frequency")
parser.add_argument("--duration", type=int, default=None, help="duration of the data")
parser.add_argument("--fmin", type=float, default=None, help="minimum frequency")
parser.add_argument("--ifos", nargs="+", default=None, help="list of detectors")

# Add injection parameters to parser
parser.add_argument("--Nt", type=int, default=100, help="Number of systems to generate")
parser.add_argument(
    "--mmin",
    type=float,
    default=None,
    help="min of masses",
)
parser.add_argument(
    "--mmax",
    type=float,
    default=None,
    help="max of masses",
)
parser.add_argument(
    "--chimin",
    type=float,
    default=None,
    help="min of dimensioless spins",
)
parser.add_argument(
    "--chimax",
    type=float,
    default=None,
    help="max of dimensioless spins",
)
parser.add_argument(
    "--distmin", type=float, default=None, help="min distance in megaparsecs"
)
parser.add_argument(
    "--distmax", type=float, default=None, help="max distance in megaparsecs"
)


# Add output parameters to parser

parser.add_argument("--output_path", type=str, default=None, help="output file path")

# parser

args = parser.parse_args()
opt = vars(args)
args = yaml.load(open(opt["config"], "r"), Loader=yaml.FullLoader)
opt.update(args)
args = opt

########################################
# Detector setup
########################################

f_sampling = args["f_sampling"]
duration = args["duration"]
fmin = args["fmin"]
ifos = args["ifos"]

freqs, psd_dict, noise_dict = generate_noise(1234, f_sampling, duration, fmin, ifos)

########################################
# Generating parameters
########################################

N = args["Nt"]
print("Generating %d systems" % (N,))

# Fix time and phase at coalesence
tc = 0.0
phic = 0.0

# Intrinsic variables
mmin = args["mmin"]
mmax = args["mmax"]
chimin = args["chimin"]
chimax = args["chimax"]
distmin = args["distmin"]
distmax = args["distmax"]

np.random.seed(1234)
# Extrinisic variable
# Change sampling to be isotropic
inclination = np.arccos(np.random.uniform(-1.0, 1.0, N))
polarization_angle = np.random.uniform(0.0, 2 * jnp.pi, N)
# Double check these angles
# ran1, ran2 = np.random.random(2 * N).reshape(2, -1)
ra = np.random.uniform(0.0, 2 * jnp.pi, N)
dec = jnp.pi / 2.0 - np.arccos(np.random.uniform(-1.0, 1.0, N))

# Finally lets generate section of parameters
# params = np.zeros([N, 11])

# for i in range(N):
#     m1 = np.random.uniform(mmin, mmax)
#     m2 = np.random.uniform(mmin, mmax)
#     chi1 = np.random.uniform(chimin, chimax)
#     chi2 = np.random.uniform(chimin, chimax)
#     dist = (distmax - distmin) * (np.random.random() ** (1 / 3)) + distmin
#     if m1 > m2:
#         Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
#     elif m1 < m2:
#         Mc, eta = ms_to_Mc_eta(jnp.array([m2, m1]))
#     params[i, 0] = Mc
#     params[i, 1] = np.clip(eta, 0.15, 0.24)
#     params[i, 2] = chi1
#     params[i, 3] = chi2
#     params[i, 4] = dist
#     params[i, 5] = tc
#     params[i, 6] = phic
#     params[i, 7] = inclination[i]
#     params[i, 8] = polarization_angle[i]
#     params[i, 9] = ra[i]
#     params[i, 10] = np.pi / 2 - dec[i]

# params = jnp.array(params)

# np.savetxt("pop_params.txt", params)
# Just for now
print("Loading data...")
params = np.loadtxt("pop_params.txt")
# print(params[0], params2[0])
# print(np.allclose(params, params2))
# quit()

########################################
# Setting up the structure to calculate Fisher Matrices
########################################

H1 = get_H1()
L1 = get_L1()
V1 = get_V1()

f_ref = fmin
# trigger_time = 1126259462.4
trigger_time = 0.0
post_trigger_duration = 2
epoch = duration - post_trigger_duration
gmst = GreenwichMeanSiderealTime(trigger_time)

# f_list = freqs[freqs > fmin]
f_list = freqs[(freqs >= fmin) & (freqs < f_sampling / 2)]
psd_list = [psd_dict["H1"], psd_dict["L1"], psd_dict["V1"]]
waveform_generator = lambda f_, theta_: gen_IMRPhenomD_polar(f_, theta_, f_ref)

strain_real_H1 = lambda params, f: get_detector_response(
    waveform_generator, params, f, H1, gmst, epoch
).real
strain_imag_H1 = lambda params, f: get_detector_response(
    waveform_generator, params, f, H1, gmst, epoch
).imag

# m1 = 30
# m2 = 25
# Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
# distance = 1600
# test_params = jnp.array(
#     [Mc, eta, 0.3, -0.4, distance, 0.0, 0.0, np.pi / 3, np.pi / 3, np.pi / 3, np.pi / 3]
# )


strain_real_L1 = lambda params, f: get_detector_response(
    waveform_generator, params, f, L1, gmst, epoch
).real
strain_imag_L1 = lambda params, f: get_detector_response(
    waveform_generator, params, f, L1, gmst, epoch
).imag


strain_real_V1 = lambda params, f: get_detector_response(
    waveform_generator, params, f, V1, gmst, epoch
).real
strain_imag_V1 = lambda params, f: get_detector_response(
    waveform_generator, params, f, V1, gmst, epoch
).imag


def get_strain_derivatives_H1(theta):
    # Lets go a bit further and try to take more general derivatives

    # hp_grad_real = vmap(grad(strain_real_H1), in_axes=(None, 0))(theta, f_list)
    # hp_grad_imag = vmap(grad(strain_imag_H1), in_axes=(None, 0))(theta, f_list)

    hp_grad_real = vmap(partial(grad(strain_real_H1), theta))(f_list)
    hp_grad_imag = vmap(partial(grad(strain_imag_H1), theta))(f_list)
    return hp_grad_real + hp_grad_imag * 1j


def get_strain_derivatives_L1(theta):
    hp_grad_real = vmap(partial(grad(strain_real_L1), theta))(f_list)
    hp_grad_imag = vmap(partial(grad(strain_imag_L1), theta))(f_list)
    return hp_grad_real + hp_grad_imag * 1j


def get_strain_derivatives_V1(theta):
    hp_grad_real = vmap(partial(grad(strain_real_V1), theta))(f_list)
    hp_grad_imag = vmap(partial(grad(strain_imag_V1), theta))(f_list)
    return hp_grad_real + hp_grad_imag * 1j


@jax.jit
def calc_SNR(p):
    h0_H1 = get_detector_response(waveform_generator, p, f_list, H1, gmst, epoch)
    h0_L1 = get_detector_response(waveform_generator, p, f_list, L1, gmst, epoch)
    h0_V1 = get_detector_response(waveform_generator, p, f_list, V1, gmst, epoch)
    SNR2_H1 = inner_product(h0_H1, h0_H1, f_list, PSD_vals_H1)
    SNR2_L1 = inner_product(h0_L1, h0_L1, f_list, PSD_vals_L1)
    SNR2_V1 = inner_product(h0_V1, h0_V1, f_list, PSD_vals_V1)

    return jnp.sqrt(SNR2_H1 + SNR2_L1 + SNR2_V1)


PSD_vals_H1 = jax.numpy.interp(f_list, freqs, psd_list[0])
PSD_vals_L1 = jax.numpy.interp(f_list, freqs, psd_list[1])
PSD_vals_V1 = jax.numpy.interp(f_list, freqs, psd_list[2])

# np.savetxt("PSD_vals_H1.txt", np.c_[f_list, PSD_vals_H1])
# np.savetxt("PSD_vals_L1.txt", np.c_[f_list, PSD_vals_L1])
# np.savetxt("PSD_vals_V1.txt", np.c_[f_list, PSD_vals_V1])
# quit()


#####################################################
# def calc_SNR_test(p):
#     h0_H1 = get_detector_response(waveform_generator, p, f_list, H1, gmst, epoch)
#     h0_L1 = get_detector_response(waveform_generator, p, f_list, L1, gmst, epoch)
#     h0_V1 = get_detector_response(waveform_generator, p, f_list, V1, gmst, epoch)
#     SNR2_H1 = inner_product(h0_H1, h0_H1, f_list, PSD_vals_H1)
#     SNR2_L1 = inner_product(h0_L1, h0_L1, f_list, PSD_vals_L1)
#     SNR2_V1 = inner_product(h0_V1, h0_V1, f_list, PSD_vals_L1)

#     return jnp.sqrt(SNR2_H1 + SNR2_L1 + SNR2_V1)


# m1 = 30
# m2 = 25
# Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
# distance = 1600
# test_params = jnp.array(
#     [Mc, eta, 0.3, -0.4, distance, 0.0, 0.0, np.pi / 3, np.pi / 3, np.pi / 3, np.pi / 3]
# )

# print(calc_SNR(test_params))
# print(get_strain_derivatives_H1(test_params))
# quit()
#####################################################

Nderivs = 11


@jax.jit
def get_Fisher_H1(p):
    strain_grad = get_strain_derivatives_H1(p)
    F = jnp.zeros([Nderivs, Nderivs])
    for i in range(Nderivs):
        for j in range(Nderivs):
            F = F.at[i, j].set(
                inner_product(strain_grad[:, i], strain_grad[:, j], f_list, PSD_vals_H1)
            )
    return F


@jax.jit
def get_Fisher_L1(p):
    strain_grad = get_strain_derivatives_L1(p)
    F = jnp.zeros([Nderivs, Nderivs])
    for i in range(Nderivs):
        for j in range(Nderivs):
            F = F.at[i, j].set(
                inner_product(strain_grad[:, i], strain_grad[:, j], f_list, PSD_vals_L1)
            )
    return F


@jax.jit
def get_Fisher_V1(p):
    strain_grad = get_strain_derivatives_V1(p)
    F = jnp.zeros([Nderivs, Nderivs])
    for i in range(Nderivs):
        for j in range(Nderivs):
            F = F.at[i, j].set(
                inner_product(strain_grad[:, i], strain_grad[:, j], f_list, PSD_vals_V1)
            )
    return F


# sky_error_list = []


# def eigsorted(cov):
#     vals, vecs = np.linalg.eigh(cov)
#     order = vals.argsort()[::-1]
#     return vals[order], vecs[:, order]


# fig, ax = plt.subplots()
# for i in range(N):
#     print(i)

#     F = get_Fisher(params[i])
#     # cov = np.linalg.inv(F)

#     # Mceta_cov = cov[:2, :2]
#     Mceta_cov = np.linalg.inv(F[:2, :2])
#     pos = params[i, :2]

#     vals, vecs = eigsorted(Mceta_cov)
#     theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

#     # Width and height are "full" widths, not radius
#     width, height = 2 * 1.0 * np.sqrt(vals)
#     ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, fill=False)
#     ax.add_artist(ellip)
#     plt.draw()

#     # cov_error = np.sqrt(cov[9, 9] * cov[10, 10] - cov[9, 10] ** 2)
#     # sky_error = -2.0 * np.pi * np.sin(dec[i]) * cov_error * (1 - np.log(90 / 100))
#     # sky_error_list.append(sky_error)


# ax.set_xlim(20, 25)
# ax.set_ylim(0.242, 0.25)
# ax.set_xlabel("Mchirp")
# ax.set_ylabel("eta")
# plt.savefig("plots/Fisher_matrices.pdf", bbox_inches="tight")

print("Starting compilation")
start = time.time()
F_H1 = get_Fisher_H1(params[0])
F_L1 = get_Fisher_L1(params[0])
F_V1 = get_Fisher_V1(params[0])
end = time.time()
print("Compile time is:", end - start)

# # Meta_error_list = []
sky_error_list = []
SNR_list = []
Mc_error = []


start = time.time()
for i in tqdm(range(N)):
    # print("Parameters:", params[i])
    SNR = calc_SNR(params[i])
    SNR_list.append(SNR)
    # print("SNR of system: %.2f" % (SNR,))
    F_H1 = get_Fisher_H1(params[i])
    F_L1 = get_Fisher_L1(params[i])
    F_V1 = get_Fisher_V1(params[i])
    F = F_H1 + F_L1 + F_V1
    cov = np.linalg.inv(F)
    # print(np.diagonal(cov))

    cov_subset = cov[-2:, -2:]
    cov_error = np.sqrt(cov_subset[0, 0] * cov_subset[1, 1] - cov_subset[0, 1] ** 2)
    cov_error *= (
        -2.0 * np.pi * np.sin(params[i, 10]) * np.log(1 - 90 / 100) * (180 / np.pi) ** 2
    )
    sky_error_list.append(cov_error)
    Mc_error.append(np.sqrt(cov[0, 0]))

end = time.time()
print("Time per Fisher calculation is:", (end - start) / N)

# print(
#     params[np.argmin(sky_error_list)],
#     SNR_list[np.argmin(sky_error_list)],
#     np.min(sky_error_list),
# )

np.savetxt("sky_localization.txt", np.c_[params, sky_error_list, SNR_list, Mc_error])

# @jax.jit
# def sky_error(p):
#     F_H1 = get_Fisher_H1(p)
#     F_L1 = get_Fisher_L1(p)
#     F_V1 = get_Fisher_V1(p)
#     F = F_H1 + F_L1 + F_V1
#     cov = jnp.linalg.inv(F)

#     cov_subset = cov[-2:, -2:]
#     cov_error = jnp.sqrt(cov_subset[0, 0] * cov_subset[1, 1] - cov_subset[0, 1] ** 2)
#     cov_error *= (
#         -2.0 * jnp.pi * jnp.sin(p[10]) * jnp.log(1 - 90 / 100) * (180 / jnp.pi) ** 2
#     )
#     return cov_error
# return F


# start = time.time()
# print(sky_error(params[0]))
# end = time.time()
# print("Compile time is:", end - start)
# start = time.time()

# start = time.time()
# sky_error_func = jax.jit(vmap(sky_error, 0))
# sky_error_func(params[:3])
# end = time.time()
# print("Compile time is:", end - start)

# start = time.time()
# sky_error_list = sky_error_func(params)
# end = time.time()
# print("Time per Fisher calculation is:", (end - start) / N)

GWbench_skypositions = np.loadtxt("GWbench_skyareas.txt")
GWbench_Mcs = np.loadtxt("GWbench_Mcs.txt")
GWbench_SNRs = np.loadtxt("GWbench_SNRs.txt")

plt.figure(figsize=(7, 5))
bins = np.geomspace(0.1, np.max(sky_error_list), 50)
plt.hist(sky_error_list, bins=bins, density=True, alpha=0.5, label="ripple")
plt.hist(GWbench_skypositions, bins=bins, density=True, alpha=0.5, label="GWbench")
plt.xscale("log")
plt.legend()
plt.xlabel("Sky Localization Error, [deg$^2$]")
plt.savefig("plots/sky_error.pdf", bbox_inches="tight")

plt.figure(figsize=(7, 5))
bins = np.geomspace(0.1, np.max(Mc_error), 50)
plt.hist(Mc_error, bins=bins, density=True, alpha=0.5, label="ripple")
plt.hist(GWbench_Mcs, bins=bins, density=True, alpha=0.5, label="GWbench")
plt.xscale("log")
plt.legend()
plt.xlabel("Mc_error Error")
plt.savefig("plots/Mc_error.pdf", bbox_inches="tight")

plt.figure(figsize=(7, 5))
bins = np.linspace(0.0, np.max(SNR_list), 50)
plt.hist(SNR_list, bins=bins, density=True, alpha=0.5, label="ripple")
plt.hist(GWbench_SNRs, bins=bins, density=True, alpha=0.5, label="GWbench")
plt.xlabel("SNR")
plt.legend()
plt.savefig("plots/SNR.pdf", bbox_inches="tight")

# plt.figure(figsize=(7, 5))
# plt.hist((Mc_error - GWbench_Mcs) / GWbench_Mcs, bins=50)
# plt.xlabel("MC_diff")
# plt.savefig("plots/MC_diff.pdf", bbox_inches="tight")

# plt.figure(figsize=(7, 5))
# plt.hist((sky_error_list - GWbench_skypositions) / GWbench_skypositions, bins=50)
# plt.xlabel("sky_error_diff")
# plt.savefig("plots/sky_error_diff.pdf", bbox_inches="tight")

# plt.figure(figsize=(7, 5))
# bins = np.linspace(0.0, np.max(Meta_error_list), 100)
# plt.hist(Meta_error_list, bins=bins)
# plt.xlabel("Chirp Mass Error")
# plt.savefig("plots/Mc_hist.pdf", bbox_inches="tight")

# ax.set_xlim(20, 25)
# ax.set_ylim(0.242, 0.25)
# ax.set_xlabel("Mchirp")
# ax.set_ylabel("eta")
# plt.savefig("plots/Fisher_matrices.pdf", bbox_inches="tight")

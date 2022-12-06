# Credit some part of the source code from bilby

import jax.numpy as jnp
from jaxgw.PE.constants import *


def make_detector_response(detector_tensor, detector_vertex):
    antenna_response_plus = make_antenna_response(detector_tensor, "plus")
    antenna_response_cross = make_antenna_response(detector_tensor, "cross")

    def detector_response(f, hp, hc, ra, dec, gmst, psi):
        output = (
            antenna_response_plus(ra, dec, gmst, psi) * hp
            + antenna_response_cross(ra, dec, gmst, psi) * hc
        )
        timeshift = time_delay_geocentric(
            detector_vertex, jnp.array([0.0, 0.0, 0.0]), ra, dec, gmst
        )
        output = output * jnp.exp(-1j * 2 * jnp.pi * f * timeshift)
        return output

    return detector_response


def get_detector_response(waveform, params, data_f, detector, gmst, epoch):
    detector_response = make_detector_response(detector[0], detector[1])
    theta_waveform = params[:8]
    theta_waveform = theta_waveform.at[5].set(0)

    hp, hc = waveform(data_f, theta_waveform)
    ra = params[9]
    dec = params[10]

    output = detector_response(data_f, hp, hc, ra, dec, gmst, params[8]) * jnp.exp(
        -1j * 2 * jnp.pi * data_f * (epoch + params[5])
    )
    return output


##########################################################
# Construction of arms
##########################################################


def construct_arm(latitude, longitude, arm_tilt, arm_azimuth):
    """

     Args:

        latitude: Latitude in radian
        longitude: Longitude in radian
        arm_tilt: Arm tilt in radian
        arm_azimuth: Arm azimuth in radian
   
    """

    e_long = jnp.array([-jnp.sin(longitude), jnp.cos(longitude), 0])
    e_lat = jnp.array(
        [
            -jnp.sin(latitude) * jnp.cos(longitude),
            -jnp.sin(latitude) * jnp.sin(longitude),
            jnp.cos(latitude),
        ]
    )
    e_h = jnp.array(
        [
            jnp.cos(latitude) * jnp.cos(longitude),
            jnp.cos(latitude) * jnp.sin(longitude),
            jnp.sin(latitude),
        ]
    )

    return (
        jnp.cos(arm_tilt) * jnp.cos(arm_azimuth) * e_long
        + jnp.cos(arm_tilt) * jnp.sin(arm_azimuth) * e_lat
        + jnp.sin(arm_tilt) * e_h
    )


def detector_tensor(arm1, arm2):
    return 0.5 * (jnp.einsum("i,j->ij", arm1, arm1) - jnp.einsum("i,j->ij", arm2, arm2))


##########################################################
# Construction of detector tensor
##########################################################


def make_get_polarization_tensor(mode):

    """

    Since most of the application will only use specific modes,
    this function hoist the if-else loop out from the actual kernel to save time from compiling the kernel.

    Args:
        mode: string

    """

    if mode.lower() == "plus":
        kernel = lambda m, n: jnp.einsum("i,j->ij", m, m) - jnp.einsum("i,j->ij", n, n)
    elif mode.lower() == "cross":
        kernel = lambda m, n: jnp.einsum("i,j->ij", m, n) + jnp.einsum("i,j->ij", n, m)
    elif mode.lower() == "breathing":
        kernel = lambda m, n: jnp.einsum("i,j->ij", m, m) + jnp.einsum("i,j->ij", n, n)

        # Calculating omega here to avoid calculation when model in [plus, cross, breathing]
        if mode.lower() == "longitudinal":

            def kernel(m, n):
                omega = jnp.cross(m, n)
                return jnp.einsum("i,j->ij", omega, omega)

        elif mode.lower() == "x":

            def kernel(m, n):
                omega = jnp.cross(m, n)
                return jnp.einsum("i,j->ij", m, omega) + jnp.einsum("i,j->ij", omega, m)

        elif mode.lower() == "y":

            def kernel(m, n):
                omega = jnp.cross(m, n)
                return jnp.einsum("i,j->ij", n, omega) + jnp.einsum("i,j->ij", omega, n)

        else:
            raise ValueError("{} not a polarization mode!".format(mode))

    def get_polarization_tensor(ra, dec, gmst, psi):
        gmst = jnp.mod(gmst, 2 * jnp.pi)
        phi = ra - gmst
        theta = jnp.pi / 2 - dec

        u = jnp.array(
            [
                jnp.cos(phi) * jnp.cos(theta),
                jnp.cos(theta) * jnp.sin(phi),
                -jnp.sin(theta),
            ]
        )
        v = jnp.array([-jnp.sin(phi), jnp.cos(phi), 0])
        m = -u * jnp.sin(psi) - v * jnp.cos(psi)
        n = -u * jnp.cos(psi) + v * jnp.sin(psi)

        return kernel(m, n)

    return get_polarization_tensor


def make_antenna_response(detector_tensor, mode):
    kernel = make_get_polarization_tensor(mode)

    def antenna_response(ra, dec, gmst, psi):
        polarization_tensor = kernel(ra, dec, gmst, psi)
        return jnp.einsum("ij,ij->", detector_tensor, polarization_tensor)

    return antenna_response


def time_delay_geocentric(detector1, detector2, ra, dec, gmst):
    """
    Calculate time delay between two detectors in geocentric coordinates based on XLALArrivaTimeDiff in TimeDelay.c

    Parameters
    ==========
    detector1: array_like
        Cartesian coordinate vector for the first detector in the geocentric frame
        generated by the Interferometer class as self.vertex.
    detector2: array_like
        Cartesian coordinate vector for the second detector in the geocentric frame.
        To get time delay from Earth center, use detector2 = np.array([0,0,0])
    ra: float
        Right ascension of the source in radians
    dec: float
        Declination of the source in radians
    gmst: float
        Greenwich mean sidereal time in radians

    Returns
    =======
    float: Time delay between the two detectors in the geocentric frame

    """
    gmst = jnp.mod(gmst, 2 * jnp.pi)
    phi = ra - gmst
    theta = jnp.pi / 2 - dec
    omega = jnp.array(
        [jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)]
    )
    delta_d = detector2 - detector1
    return jnp.dot(omega, delta_d) / speed_of_light


def get_vertex_position_geocentric(latitude, longitude, elevation):
    """
    Calculate the position of the IFO vertex in geocentric coordinates in meters.

    Based on arXiv:gr-qc/0008066 Eqs. B11-B13 except for the typo in the definition of the local radius.
    See Section 2.1 of LIGO-T980044-10 for the correct expression

    Parameters
    ==========
    latitude: float
        Latitude in radians
    longitude:
        Longitude in radians
    elevation:
        Elevation in meters

    Returns
    =======
    array_like: A 3D representation of the geocentric vertex position

    """
    semi_major_axis = 6378137  # for ellipsoid model of Earth, in m
    semi_minor_axis = 6356752.314  # in m
    radius = semi_major_axis ** 2 * (
        semi_major_axis ** 2 * jnp.cos(latitude) ** 2
        + semi_minor_axis ** 2 * jnp.sin(latitude) ** 2
    ) ** (-0.5)
    x_comp = (radius + elevation) * jnp.cos(latitude) * jnp.cos(longitude)
    y_comp = (radius + elevation) * jnp.cos(latitude) * jnp.sin(longitude)
    z_comp = ((semi_minor_axis / semi_major_axis) ** 2 * radius + elevation) * jnp.sin(
        latitude
    )
    return jnp.array([x_comp, y_comp, z_comp])


# import numpy as np
# import jax.numpy as jnp

# class TestTransform:
#     def test_sky_location_transform(self):
#         from bilby.gw.utils import zenith_azimuth_to_ra_dec as bilby_earth_to_sky
#         from bilby.gw.detector.networks import InterferometerList
        
#         from jimgw.single_event.utils import zenith_azimuth_to_ra_dec as jimgw_earth_to_sky
#         from jimgw.single_event.detector import detector_preset
#         from astropy.time import Time

#         ifos = ["H1", "L1"]
#         geocent_time = 1000000000

#         import matplotlib.pyplot as plt

#         for zenith in np.linspace(0, np.pi, 10):
#             for azimuth in np.linspace(0, 2*np.pi, 10):
#                 bilby_sky_location = np.array(bilby_earth_to_sky(zenith, azimuth, geocent_time, InterferometerList(ifos)))
#                 jimgw_sky_location = np.array(jimgw_earth_to_sky(zenith, azimuth, Time(geocent_time, format="gps").sidereal_time("apparent", "greenwich").rad, detector_preset[ifos[0]].vertex - detector_preset[ifos[1]].vertex))
#                 assert np.allclose(bilby_sky_location, jimgw_sky_location, atol=1e-4)

#     def test_spin_transform(self):
#         from bilby.gw.conversion import bilby_to_lalsimulation_spins as bilby_spin_transform
#         from bilby.gw.conversion import symmetric_mass_ratio_to_mass_ratio, chirp_mass_and_mass_ratio_to_component_masses

#         from jimgw.single_event.utils import spin_to_cartesian_spin as jimgw_spin_transform

#         for _ in range(100):
#             thetaJN = jnp.array(np.random.uniform(0, np.pi))
#             phiJL = jnp.array(np.random.uniform(0, np.pi))
#             theta1 = jnp.array(np.random.uniform(0, np.pi))
#             theta2 = jnp.array(np.random.uniform(0, np.pi))
#             phi12 = jnp.array(np.random.uniform(0, np.pi))
#             chi1 = jnp.array(np.random.uniform(0, 1))
#             chi2 = jnp.array(np.random.uniform(0, 1))
#             M_c = jnp.array(np.random.uniform(1, 100))
#             eta = jnp.array(np.random.uniform(0.1, 0.25))
#             fRef = jnp.array(np.random.uniform(10, 1000))
#             phiRef = jnp.array(np.random.uniform(0, 2*np.pi))

#             q = symmetric_mass_ratio_to_mass_ratio(eta)
#             m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(M_c, q)
#             MsunInkg = 1.9884e30
#             bilby_spin = jnp.array(bilby_spin_transform(thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2, m1*MsunInkg, m2*MsunInkg, fRef, phiRef))
#             jimgw_spin = jnp.array(jimgw_spin_transform(thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2, M_c, eta, fRef, phiRef))
#             assert np.allclose(bilby_spin, jimgw_spin, atol=1e-4)

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import requests
from gwpy.timeseries import TimeSeries
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped
from beartype import beartype as typechecker
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey
from . import data as jd
from typing import Optional

from jimgw.constants import C_SI, EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS
from jimgw.single_event.wave import Polarization
import logging

DEG_TO_RAD = jnp.pi / 180

# TODO: Need to expand this list. Currently it is only O3.
asd_file_dict = {
    "H1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt",
    "L1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt",
    "V1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-V1_sensitivity_strain_asd.txt",
}

_DEF_GWPY_KWARGS = {"cache": True}


class Detector(ABC):
    """Base class for all detectors.
    """

    name: str

    # NOTE: for some detectors (e.g. LISA, ET) data could be a list of Data
    # objects so this might be worth revisiting
    data: jd.Data
    psd: jd.PowerSpectrum

    frequency_bounds: tuple[float, float] = (0., float("inf"))

    @abstractmethod
    def fd_response(
        self,
        frequency: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict,
        **kwargs,
    ) -> Float[Array, " n_sample"]:
        """Modulate the waveform in the sky frame by the detector response
        in the frequency domain.
        """
        pass

    @abstractmethod
    def td_response(
        self,
        time: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict,
        **kwargs,
    ) -> Float[Array, " n_sample"]:
        """Modulate the waveform in the sky frame by the detector response
        in the time domain.
        """
        pass

    def set_frequency_bounds(self, f_min: Optional[float] = None,
                             f_max: Optional[float] = None) -> None:
        """Set the frequency bounds for the detector.

        Parameters
        ----------
        f_min : float
            Minimum frequency.
        f_max : float
            Maximum frequency.
        """
        bounds = list(self.frequency_bounds)
        if f_min is not None:
            bounds[0] = f_min
        if f_max is not None:
            bounds[1] = f_max
        self.frequency_bounds = tuple(bounds)  # type: ignore

    @property
    def fd_data_slice(self):
        return self.data.frequency_slice(*self.frequency_bounds)

    @property
    def psd_slice(self):
        return self.psd.frequency_slice(*self.frequency_bounds)


class GroundBased2G(Detector):
    """Object representing a ground-based detector. Contains information
    about the location and orientation of the detector on Earth, as well as
    actual strain data and the PSD of the associated noise.

    Attributes
    ----------
    name : str
        Name of the detector.
    latitude : Float
        Latitude of the detector in radians.
    longitude : Float
        Longitude of the detector in radians.
    xarm_azimuth : Float
        Azimuth of the x-arm in radians.
    yarm_azimuth : Float
        Azimuth of the y-arm in radians.
    xarm_tilt : Float
        Tilt of the x-arm in radians.
    yarm_tilt : Float
        Tilt of the y-arm in radians.
    elevation : Float
        Elevation of the detector in meters.
    polarization_mode : list[Polarization]
        List of polarization modes (`pc` for plus and cross) to be used in
        computing antenna patterns; in the future, this could be expanded to
        include non-GR modes.
    frequencies : Float[Array, " n_sample"]
        Array of Fourier frequencies.
    data : Float[Array, " n_sample"]
        Array of Fourier-domain strain data.
    psd : Float[Array, " n_sample"]
        Array of noise power spectral density.
    """
    polarization_mode: list[Polarization]
    data: jd.Data
    psd: jd.PowerSpectrum

    latitude: Float = 0
    longitude: Float = 0
    xarm_azimuth: Float = 0
    yarm_azimuth: Float = 0
    xarm_tilt: Float = 0
    yarm_tilt: Float = 0
    elevation: Float = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __init__(self, name: str, latitude: float = 0, longitude: float = 0,
                 elevation: float = 0, xarm_azimuth: float = 0,
                 yarm_azimuth: float = 0, xarm_tilt: float = 0,
                 yarm_tilt: float = 0, modes: str = "pc"):
        self.name = name

        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.xarm_azimuth = xarm_azimuth
        self.yarm_azimuth = yarm_azimuth
        self.xarm_tilt = xarm_tilt
        self.yarm_tilt = yarm_tilt

        self.polarization_mode = [Polarization(m) for m in modes]
        self.data = jd.Data()
        self.psd = jd.PowerSpectrum()

    @staticmethod
    def _get_arm(
        lat: Float, lon: Float, tilt: Float, azimuth: Float
    ) -> Float[Array, " 3"]:
        """Construct detector-arm vectors in geocentric Cartesian coordinates.

        Parameters
        ---------
        lat : Float
            vertex latitude in rad.
        lon : Float
            vertex longitude in rad.
        tilt : Float
            arm tilt in rad.
        azimuth : Float
            arm azimuth in rad.

        Returns
        -------
        arm : Float[Array, " 3"]
            detector arm vector in geocentric Cartesian coordinates.
        """
        e_lon = jnp.array([-jnp.sin(lon), jnp.cos(lon), 0])
        e_lat = jnp.array(
            [-jnp.sin(lat) * jnp.cos(lon), -jnp.sin(lat)
             * jnp.sin(lon), jnp.cos(lat)]
        )
        e_h = jnp.array(
            [jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat)
             * jnp.sin(lon), jnp.sin(lat)]
        )

        return (
            jnp.cos(tilt) * jnp.cos(azimuth) * e_lon
            + jnp.cos(tilt) * jnp.sin(azimuth) * e_lat
            + jnp.sin(tilt) * e_h
        )

    @property
    def arms(self) -> tuple[Float[Array, " 3"], Float[Array, " 3"]]:
        """Detector arm vectors (x, y).

        Returns
        -------
        x : Float[Array, " 3"]
            x-arm vector.
        y : Float[Array, " 3"]
            y-arm vector.
        """
        x = self._get_arm(
            self.latitude, self.longitude, self.xarm_tilt, self.xarm_azimuth
        )
        y = self._get_arm(
            self.latitude, self.longitude, self.yarm_tilt, self.yarm_azimuth
        )
        return x, y

    @property
    def tensor(self) -> Float[Array, " 3 3"]:
        """Detector tensor defining the strain measurement.

        For a 2-arm differential-length detector, this is given by:

        .. math::

            D_{ij} = \\left(x_i x_j - y_i y_j\\right)/2

        for unit vectors :math:`x` and :math:`y` along the x and y arms.

        Returns
        -------
        tensor : Float[Array, " 3 3"]
            detector tensor.
        """
        # TODO: this could easily be generalized for other detector geometries
        arm1, arm2 = self.arms
        return 0.5 * jnp.einsum("i,j->ij", arm1, arm1) - \
            jnp.einsum("i,j->ij", arm2, arm2)

    @property
    def vertex(self) -> Float[Array, " 3"]:
        """Detector vertex coordinates in the reference celestial frame. Based
        on arXiv:gr-qc/0008066 Eqs. (B11-B13) except for a typo in the
        definition of the local radius; see Section 2.1 of LIGO-T980044-10.

        Returns
        -------
        vertex : Float[Array, " 3"]
            detector vertex coordinates.
        """
        # get detector and Earth parameters
        lat = self.latitude
        lon = self.longitude
        h = self.elevation
        major, minor = EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS
        # compute vertex location
        r = major**2 * (
            major**2 * jnp.cos(lat) ** 2 + minor**2 * jnp.sin(lat) ** 2
        ) ** (-0.5)
        x = (r + h) * jnp.cos(lat) * jnp.cos(lon)
        y = (r + h) * jnp.cos(lat) * jnp.sin(lon)
        z = ((minor / major) ** 2 * r + h) * jnp.sin(lat)
        return jnp.array([x, y, z])

    def fd_response(
        self,
        frequency: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict[str, Float],
        **kwargs,
    ) -> Array:
        """
        Modulate the waveform in the sky frame by the detector response in the frequency domain.
        """
        ra, dec, psi, gmst = params["ra"], params["dec"], params["psi"], params["gmst"]
        antenna_pattern = self.antenna_pattern(ra, dec, psi, gmst)
        timeshift = self.delay_from_geocenter(ra, dec, gmst)
        h_detector = jax.tree_util.tree_map(
            lambda h, antenna: h
            * antenna
            * jnp.exp(-2j * jnp.pi * frequency * timeshift),
            h_sky,
            antenna_pattern,
        )
        return jnp.sum(jnp.stack(jax.tree_util.tree_leaves(h_detector)), axis=0)

    def td_response(
        self,
        time: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict,
        **kwargs,
    ) -> Array:
        """
        Modulate the waveform in the sky frame by the detector response in the time domain.
        """
        raise NotImplementedError

    def delay_from_geocenter(self, ra: Float, dec: Float, gmst: Float) -> Float:
        """
        Calculate time delay between two detectors in geocentric
        coordinates based on XLALArrivaTimeDiff in TimeDelay.c

        https://lscsoft.docs.ligo.org/lalsuite/lal/group___time_delay__h.html

        Parameters
        ---------
        ra : Float
            right ascension of the source in rad.
        dec : Float
            declination of the source in rad.
        gmst : Float
            Greenwich mean sidereal time in rad.

        Returns
        -------
        Float: time delay from Earth center.
        """
        delta_d = -self.vertex
        gmst = jnp.mod(gmst, 2 * jnp.pi)
        phi = ra - gmst
        theta = jnp.pi / 2 - dec
        omega = jnp.array(
            [
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta),
            ]
        )
        return jnp.dot(omega, delta_d) / C_SI

    def antenna_pattern(self, ra: Float, dec: Float, psi: Float, gmst: Float) -> dict:
        """
        Computes {name} antenna patterns for {modes} polarizations
        at the specified sky location, orientation and GMST.

        In the long-wavelength approximation, the antenna pattern for a
        given polarization is the dyadic product between the detector
        tensor and the corresponding polarization tensor.

        Parameters
        ---------
        ra : Float
            source right ascension in radians.
        dec : Float
            source declination in radians.
        psi : Float
            source polarization angle in radians.
        gmst : Float
            Greenwich mean sidereal time (GMST) in radians.
        modes : str
            string of polarizations to include, defaults to tensor modes: 'pc'.

        Returns
        -------
        result : list
            antenna pattern values for {modes}.
        """
        detector_tensor = self.tensor

        antenna_patterns = {}
        for polarization in self.polarization_mode:
            wave_tensor = polarization.tensor_from_sky(ra, dec, psi, gmst)
            antenna_patterns[polarization.name] = jnp.einsum(
                "ij,ij->", detector_tensor, wave_tensor
            )

        return antenna_patterns

    @jaxtyped(typechecker=typechecker)
    def load_psd(
        self, freqs: Float[Array, " n_sample"], psd_file: str = ""
    ) -> Float[Array, " n_sample"]:
        if psd_file == "":
            print("Grabbing GWTC-2 PSD for " + self.name)
            url = asd_file_dict[self.name]
            data = requests.get(url)
            open(self.name + ".txt", "wb").write(data.content)
            f, asd_vals = np.loadtxt(self.name + ".txt", unpack=True)
            psd_vals = asd_vals**2
        else:
            f, psd_vals = np.loadtxt(psd_file, unpack=True)

        psd = interp1d(f, psd_vals, fill_value=(
            psd_vals[0], psd_vals[-1]))(freqs)  # type: ignore
        psd = jnp.array(psd)
        return psd

    def set_data(self, data: jd.Data | Array, **kws) -> None:
        """Add data to detector.

        Arguments
        ---------
        data : jd.Data | Array
            Data to be added to the detector, either as a `jd.Data` object
            or as a timeseries array.
        kws : dict
            Additional keyword arguments to pass to `jd.Data` constructor.
        """
        if isinstance(data, jd.Data):
            self.data = data
        else:
            self.data = jd.Data(data, **kws)

    def set_psd(self, psd: jd.PowerSpectrum | Array, **kws) -> None:
        """Add PSD to detector.

        Arguments
        ---------
        psd : jd.PowerSpectrum | Array
            PSD to be added to the detector, either as a `jd.PowerSpectrum`
            object or as a timeseries array.
        kws : dict
            Additional keyword arguments to pass to `jd.PowerSpectrum`
            constructor.
        """
        if isinstance(psd, jd.PowerSpectrum):
            self.psd = psd
        else:
            # not clear if we want to support this
            self.psd = jd.PowerSpectrum(psd, **kws)


H1 = GroundBased2G(
    "H1",
    latitude=(46 + 27.0 / 60 + 18.528 / 3600) * DEG_TO_RAD,
    longitude=-(119 + 24.0 / 60 + 27.5657 / 3600) * DEG_TO_RAD,
    xarm_azimuth=125.9994 * DEG_TO_RAD,
    yarm_azimuth=215.9994 * DEG_TO_RAD,
    xarm_tilt=-6.195e-4,
    yarm_tilt=1.25e-5,
    elevation=142.554,
    modes="pc",
)

L1 = GroundBased2G(
    "L1",
    latitude=(30 + 33.0 / 60 + 46.4196 / 3600) * DEG_TO_RAD,
    longitude=-(90 + 46.0 / 60 + 27.2654 / 3600) * DEG_TO_RAD,
    xarm_azimuth=197.7165 * DEG_TO_RAD,
    yarm_azimuth=287.7165 * DEG_TO_RAD,
    xarm_tilt=0,
    yarm_tilt=0,
    elevation=-6.574,
    modes="pc",
)

V1 = GroundBased2G(
    "V1",
    latitude=(43 + 37.0 / 60 + 53.0921 / 3600) * DEG_TO_RAD,
    longitude=(10 + 30.0 / 60 + 16.1887 / 3600) * DEG_TO_RAD,
    xarm_azimuth=70.5674 * DEG_TO_RAD,
    yarm_azimuth=160.5674 * DEG_TO_RAD,
    xarm_tilt=0,
    yarm_tilt=0,
    elevation=51.884,
    modes="pc",
)

detector_preset = {
    "H1": H1,
    "L1": L1,
    "V1": V1,
}

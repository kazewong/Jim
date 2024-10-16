from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import requests
from gwpy.timeseries import TimeSeries
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped, Complex
from typing import Optional, Any
from beartype import beartype as typechecker
from scipy.interpolate import interp1d
import scipy.signal as sig
from scipy.signal.windows import tukey

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


class Data(ABC):
    """
    Base class for all data. The time domain data are considered the primary
    entitiy; the Fourier domain data are derived from an FFT after applying a
    window. The structure is set up so that :attr:`td` and :attr:`fd` are always
    Fourier conjugates of each other: the one-sided Fourier series is complete
    up to the Nyquist frequency

    """
    name: str

    td: Float[Array, " n_time"]
    fd: Complex[Array, " n_time // 2 + 1"]

    epoch: float
    delta_t: float

    window: Float[Array, " n_time"]

    @property
    def n_time(self) -> int:
        """Number of time samples."""
        return len(self.td)

    @property
    def n_freq(self) -> int:
        """Number of frequency samples."""
        return self.n_time // 2 + 1

    @property
    def duration(self) -> float:
        """Duration of the data in seconds."""
        return self.n_time * self.delta_t

    @property
    def sampling_frequency(self) -> float:
        """Sampling frequency of the data in Hz."""
        return 1 / self.delta_t

    @property
    def times(self) -> Float[Array, " n_time"]:
        """Times of the data in seconds."""
        return jnp.arange(self.n_time) * self.delta_t + self.epoch

    @property
    def frequencies(self) -> Float[Array, " n_time // 2 + 1"]:
        """Frequencies of the data in Hz."""
        return jnp.fft.rfftfreq(self.n_time, self.delta_t)

    @property
    def has_fd(self) -> bool:
        """Whether the Fourier domain data has been computed."""
        return bool(np.any(self.fd))

    def __init__(self, td: Float[Array, " n_time"],
                 delta_t: float,
                 epoch: float = 0.,
                 name: Optional[str] = None,
                 window: Optional[Float[Array, " n_time"]] = None)\
            -> None:
        """Initialize the data class.

        Arguments
        ---------
        td: array
            Time domain data
        delta_t: float
            Time step of the data in seconds.
        epoch: float, optional
            Epoch of the data in seconds (default: 0)
        name: str, optional
            Name of the data (default: '')
        window: array, optional
            Window function to apply to the data before FFT (default: None)
        """
        self.td = td
        self.fd = jnp.zeros(self.n_freq)
        self.delta_t = delta_t
        self.epoch = epoch
        if window is None:
            self.window = jnp.ones_like(self.td)
        else:
            self.window = window
        self.name = name or ''

    def set_tukey_window(self, alpha: float = 0.2) -> None:
        """Create a Tukey window on the data; the window is stored in the
        :attr:`window` attribute and only applied when FFTing the data.

        Arguments
        ---------
        alpha: float, optional
            Shape parameter of the Tukey window (default: 0.2); this is
            the fraction of the segment that is tapered on each side.
        """
        logging.info(f"Setting Tukey window to {self.name} data")
        self.window = jnp.array(tukey(self.n_time, alpha))

    def fft(self, window: Optional[Float[Array, " n_time"]] = None) -> None:
        """Compute the Fourier transform of the data and store it
        in the :attr:`fd` attribute.

        Arguments
        ---------
        **kws: dict, optional
            Keyword arguments for the FFT; defaults to
        """
        logging.info(f"Computing FFT of {self.name} data")
        if window is not None:
            self.window = window
        self.fd = jnp.fft.rfft(self.td * self.window) * self.delta_t

    def frequency_slice(self, f_min: float, f_max: float) -> \
            tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        """Slice the data in the frequency domain.

        Arguments
        ---------
        f_min: float
            Minimum frequency of the slice in Hz.
        f_max: float
            Maximum frequency of the slice in Hz.

        Returns
        -------
        fd_slice: array
            Sliced data in the frequency domain.
        f_slice: array
            Frequencies of the sliced data.
        """
        f = self.frequencies
        return self.fd[(f >= f_min) & (f <= f_max)], \
            f[(f >= f_min) & (f <= f_max)]

    def to_psd(self, **kws) -> "PowerSpectrum":
        """Compute a Welch estimate of the power spectral density of the data.

        Arguments
        ---------
        **kws: dict, optional
            Keyword arguments for `scipy.signal.welch`

        Returns
        -------
        psd: PowerSpectrum
            Power spectral density of the data.
        """
        if not self.has_fd:
            self.fft()
        freq, psd = sig.welch(self.td, fs=self.sampling_frequency, **kws)
        return PowerSpectrum(jnp.array(psd), freq, self.name)

    @classmethod
    def from_gwosc(cls,
                   ifo: str,
                   gps_start_time: Float,
                   gps_end_time: Float,
                   cache: bool = True,
                   **kws) -> "Data":
        """Pull data from GWOSC.

        Arguments
        ---------
        ifo: str
            Interferometer name
        gps_start_time: float
            GPS start time of the data
        gps_end_time: float
            GPS end time of the data
        cache: bool, optional
            Whether to cache the data (default: True)
        **kws: dict, optional
            Keyword arguments for `gwpy.timeseries.TimeSeries.fetch_open_data`
            defaults to {}
        """
        duration = gps_end_time - gps_start_time
        logging.info(f"Fetching {duration} s of {ifo} data from GWOSC "
                     f"[{gps_start_time}, {gps_end_time}]")

        data_td = TimeSeries.fetch_open_data(ifo, gps_start_time, gps_end_time,
                                             cache=cache, **kws)
        return cls(data_td.value, data_td.dt.value, data_td.epoch.value, ifo)

    from_gwosc.__doc__ = from_gwosc.__doc__.format(_DEF_GWPY_KWARGS)


class PowerSpectrum(ABC):
    name: str
    values: Float[Array, " n_freq"]
    frequencies: Float[Array, " n_freq"]

    @property
    def n_freq(self) -> int:
        """Number of frequency samples."""
        return len(self.values)

    @property
    def delta_f(self) -> Float:
        """Frequency resolution of the data in Hz."""
        return self.frequencies[1] - self.frequencies[0]

    @property
    def duration(self) -> Float:
        """Duration of the data in seconds."""
        return 1 / self.delta_f

    def __init__(self, values: Float[Array, " n_freq"],
                 frequencies: Float[Array, " n_freq"],
                 name: Optional[str] = None) -> None:
        self.values = values
        self.frequencies = frequencies
        self.name = name or ''

    def slice(self, f_min: float, f_max: float) -> \
        tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        """Slice the power spectrum.

        Arguments
        ---------
        f_min: float
            Minimum frequency of the slice in Hz.
        f_max: float
            Maximum frequency of the slice in Hz.

        Returns
        -------
        psd_slice: PowerSpectrum
            Sliced power spectrum.
        """
        values = self.values[(self.frequencies >= f_min) &
                             (self.frequencies <= f_max)]
        frequencies = self.frequencies[(self.frequencies >= f_min) &
                                       (self.frequencies <= f_max)]
        return values, frequencies

    def interpolate(self, f: Float[Array, " n_sample"]) -> "PowerSpectrum":
        """Interpolate the power spectrum to a new set of frequencies.

        Arguments
        ---------
        f: array
            Frequencies to interpolate the power spectrum to.

        Returns
        -------
        psd_interp: array
            Interpolated power spectrum.
        """
        interp = interp1d(self.frequencies, self.values, kind='cubic')
        return PowerSpectrum(interp(f), f, self.name)

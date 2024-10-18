__include__ = ["Data", "PowerSpectrum"]

from abc import ABC

import jax.numpy as jnp
import numpy as np
from gwpy.timeseries import TimeSeries
from jaxtyping import Array, Float, Complex, PRNGKeyArray
from typing import Optional
# from beartype import beartype as typechecker
from scipy.interpolate import interp1d
import scipy.signal as sig
from scipy.signal.windows import tukey
import logging
import jax


DEG_TO_RAD = jnp.pi / 180

# TODO: Need to expand this list. Currently it is only O3.
asd_file_dict = {
    "H1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt",  # noqa: E501
    "L1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt",  # noqa: E501
    "V1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-V1_sensitivity_strain_asd.txt",  # noqa: E501
}


class Data(ABC):
    """Base class for all data. The time domain data are considered the primary
    entity; the Fourier domain data are derived from an FFT after applying a
    window. The structure is set up so that :attr:`td` and :attr:`fd` are
    always Fourier conjugates of each other: the one-sided Fourier series is
    complete up to the Nyquist frequency.
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

    def __init__(self, td: Float[Array, " n_time"] = jnp.array([]),
                 delta_t: float = 0.,
                 epoch: float = 0.,
                 name: str = '',
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
        self.name = name or ''
        self.td = td
        self.fd = jnp.zeros(self.n_freq)
        self.delta_t = delta_t
        self.epoch = epoch
        if window is None:
            self.set_tukey_window()
        else:
            self.window = window

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', " + \
            f"delta_t={self.delta_t}, epoch={self.epoch})"

    def __bool__(self) -> bool:
        """Check if the data is empty."""
        return len(self.td) > 0

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
        """
        duration = gps_end_time - gps_start_time
        logging.info(f"Fetching {duration} s of {ifo} data from GWOSC "
                     f"[{gps_start_time}, {gps_end_time}]")

        data_td = TimeSeries.fetch_open_data(ifo, gps_start_time, gps_end_time,
                                             cache=cache, **kws)
        return cls(data_td.value, data_td.dt.value, data_td.epoch.value, ifo)  # type: ignore # noqa: E501


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

    @property
    def sampling_frequency(self) -> Float:
        """Sampling frequency of the data in Hz."""
        return self.frequencies[-1] * 2

    def __init__(self, values: Float[Array, " n_freq"] = jnp.array([]),
                 frequencies: Float[Array, " n_freq"] = jnp.array([]),
                 name: Optional[str] = None) -> None:
        self.values = values
        self.frequencies = frequencies
        assert len(self.values) == len(self.frequencies), \
            "Values and frequencies must have the same length"
        self.name = name or ''

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', " + \
            f"frequencies={self.frequencies})"

    def __bool__(self) -> bool:
        """Check if the power spectrum is empty."""
        return len(self.values) > 0

    def frequency_slice(self, f_min: float, f_max: float) -> \
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
        mask = (self.frequencies >= f_min) & (self.frequencies <= f_max)
        return self.values[mask], self.frequencies[mask]

    def interpolate(self, f: Float[Array, " n_sample"],
                    kind: str = 'cubic', **kws) -> "PowerSpectrum":
        """Interpolate the power spectrum to a new set of frequencies.

        Arguments
        ---------
        f: array
            Frequencies to interpolate the power spectrum to.
        kind: str, optional
            Interpolation method (default: 'cubic')
        **kws: dict, optional
            Keyword arguments for `scipy.interpolate.interp1d`

        Returns
        -------
        psd_interp: array
            Interpolated power spectrum.
        """
        interp = interp1d(self.frequencies, self.values, kind=kind, **kws)
        return PowerSpectrum(interp(f), f, self.name)

    def simulate_data(
        self,
        key: PRNGKeyArray,
        # freqs: Float[Array, " n_sample"],
        # h_sky: dict[str, Float[Array, " n_sample"]],
        # params: dict[str, Float],
        # psd_file: str = "",
    ) -> Complex[Array, " n_sample"]:
        """
        Inject a signal into the detector data.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX PRNG key.
        h_sky : dict[str, Float[Array, " n_sample"]]
            Array of waveforms in the sky frame. The key is the polarization
            mode.
        params : dict[str, Float]
            Dictionary of parameters.
        psd_file : str
            Path to the PSD file.

        Returns
        -------
        None
        """
        key, subkey = jax.random.split(key, 2)
        var = self.values / (4 * self.delta_f)
        noise_real = jax.random.normal(key, shape=var.shape) * jnp.sqrt(var)
        noise_imag = jax.random.normal(subkey, shape=var.shape) * jnp.sqrt(var)
        return noise_real + 1j * noise_imag

        # WIP: this should be moved to Detector class
        
        # align_time = jnp.exp(
        #     -1j * 2 * jnp.pi * freqs * (params["epoch"] + params["t_c"])
        # )
        # signal = self.fd_response(freqs, h_sky, params) * align_time
        # self.data = signal + noise_real + 1j * noise_imag

        # # also calculate the optimal SNR and match filter SNR
        # optimal_SNR = jnp.sqrt(jnp.sum(signal * signal.conj() / var).real)
        # match_filter_SNR = jnp.sum(self.data * signal.conj() / var) / optimal_SNR

        # print(f"For detector {self.name}:")
        # print(f"The injected optimal SNR is {optimal_SNR}")
        # print(f"The injected match filter SNR is {match_filter_SNR}")

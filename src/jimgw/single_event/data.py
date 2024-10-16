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
    fd: Float[Array, " n_time // 2 + 1"]

    epoch: float
    delta_t: float

    window: Float[Array, " n_time"]

    @property
    def duration(self) -> Float:
        """Duration of the data in seconds."""
        return self.n_time * self.delta_t

    @property
    def n_time(self) -> int:
        """Number of time samples."""
        return len(self.td)

    @property
    def n_freq(self) -> int:
        """Number of frequency samples."""
        return self.n_time // 2 + 1

    @property
    def times(self) -> Float[Array, " n_time"]:
        """Times of the data in seconds."""
        return jnp.arange(self.n_time) * self.delta_t + self.epoch

    @property
    def frequencies(self) -> Float[Array, " n_time // 2 + 1"]:
        """Frequencies of the data in Hz."""
        return jnp.fft.rfftfreq(self.n_time, self.delta_t)

    def __init__(self, td: Float[Array, " n_time"],
                 delta_t: float = 1.,
                 epoch: float = 0.,
                 **kws) -> None:
        self.td = td
        self.delta_t = delta_t
        self.epoch = epoch
        self.window = kws.get("window", jnp.ones_like(self.td))

    def set_tukey_window(self, alpha: float = 0.4):
        self.window = jnp.array(tukey(self.n_time, alpha))

    def fft(self, **kws) -> None:
        if "window" in kws:
            self.window = kws["window"]
        self.fd = jnp.fft.rfft(self.td * self.window) * self.delta_t

    def frequency_slice(self, f_min: float, f_max: float) -> \
            Float[Array, " n_sample"]:
        f = self.frequencies
        return self.fd[(f >= f_min) & (f <= f_max)]

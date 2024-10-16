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
    Base class for all data.

    """

    name: str

    fd: Float[Array, " n_sample"]
    td: Float[Array, " n_sample//2"]  # fix this
    psd: Float[Array, " n_sample//2"]  # fix this
    frequencies: Float[Array, " n_sample//2"]
    times: Float[Array, " n_sample"]

    @property
    def duration(self) -> Float:
        """Duration of the data in seconds."""
        if len(self.frequencies) == 0:
            return 0
        return 1 / (self.frequencies[1] - self.frequencies[0])
    
    @property
    def delta_t(self) -> Float:
        """Sampling interval of the data in seconds."""
        return self.times[1] - self.times[0]
    
    def 

    def load_psd_from_data(self, data: TimeSeries, **kws) -> None:
        seglen = self.duration
        self.psd = data.psd(fftlength=seglen).value

    def compute_psd_from_gwosc(self,
                    start_time: Float | None = None,
                    end_time: Float | None = None,
                    off_source: bool = True,
                    **kws) -> None:
        if data is None:
            if pad:
                # pull more data to compute a PSD
                
        # n = len(data)
        # delta_t = 1.0
        # data = jnp.fft.rfft(data * tukey(n, tukey_alpha)) * delta_t
        # freq = jnp.fft.rfftfreq(n, delta_t)
        # return jnp.abs(data) ** 2 / delta_t
        raise NotImplementedError

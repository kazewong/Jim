from abc import ABC, abstractmethod
import jax.numpy as jnp
from jaxtyping import Float

class PopulationModelBase(ABC):
    @abstractmethod
    def __init__(self, *params):
        self.params = params  
        
    @abstractmethod
    def evaluate(self, pop_params: dict, data: dict) -> Float:
        """
        Evaluate the likelihood for a given set of parameters.
        """
        raise NotImplementedError


class TruncatedPowerLawModel(PopulationModelBase):
    def __init__(self):
        super().__init__()  

    def truncated_power_law(self, x, x_min, x_max, alpha):
        valid_indices = (x >= x_min) & (x <= x_max)
        C = (1 - alpha) / (x_max**(1 - alpha) - x_min**(1 - alpha))
        pdf = jnp.zeros_like(x)  
        pdf = jnp.where(valid_indices, C / (x ** alpha), pdf)
        return pdf

    def evaluate(self, pop_params: dict[str, Float], data: dict) -> Float:
        return self.truncated_power_law(data, pop_params["m_min"], pop_params["m_max"],pop_params["alpha"])
    
    
    
 


import jax
import jax.numpy as jnp
from jaxtyping import Float

from jimgw.base import LikelihoodBase
from jimgw.population.utils import extract_data_from_npz_files

class PopulationLikelihood(LikelihoodBase):
    def __init__(self, data_dir, column_name, num_samples, model_class):
        self.posteriors = extract_data_from_npz_files(data_dir, column_name, num_samples, random_seed=42)
        self.population_model = model_class()

    def evaluate(self, pop_params: dict[str, Float], data: dict) -> Float:
        model_output = self.population_model.evaluate(pop_params, self.posteriors)
        log_likelihood = jnp.sum(jnp.log(jnp.mean(model_output, axis=1)))
        return log_likelihood
    

        
        
        

    
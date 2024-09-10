import importlib
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jimgw.single_event.utils import Mc_eta_to_m1_m2
import glob 

def create_model(model_name):
    try:
        module = importlib.import_module('population_model')

        # Check if model_name is a string
        if not isinstance(model_name, str):
            raise ValueError("model_name must be a string")

        model_class = getattr(module, model_name)
        return model_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import model '{model_name}': {str(e)}")


def extract_data_from_npz_files(data_dir, column_name, num_samples=50, random_seed=42):
    """
    Extracts specified column data from the given .npz files.

    Parameters:
    - data_dir (str): The directory containing all the .npz files.
    - column_name (str): The name of the column to extract from the DataFrame.
    - num_samples (int): Number of samples to extract from each file.
    - random_seed (int): Seed for random number generation.

    Returns:
    - jnp.array: Stacked array of extracted data.
    """
    
    npz_files = glob.glob(f"{data_dir}/*.npz")
    key = jax.random.PRNGKey(random_seed)
    result_list = []

    for npz_file in npz_files:
        print(f"Loading file: {npz_file}")

        with np.load(npz_file, allow_pickle=True) as data:
            data_dict = data['arr_0'].item() 
            if column_name not in data_dict:
                raise ValueError(f"Column '{column_name}' not found in the data.")
            
            extracted_data = data_dict[column_name].reshape(-1,)

            if isinstance(extracted_data, np.ndarray):
                extracted_data = jax.device_put(extracted_data) 

            key, subkey = jax.random.split(key)
            sample_indices = jax.random.choice(subkey, extracted_data.shape[0], shape=(num_samples,), replace=True)

            sampled_data = extracted_data[sample_indices]
            result_list.append(sampled_data)
    stacked_array = jnp.stack(result_list)
    
    return stacked_array
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"

import numpy as np
import matplotlib.pyplot as plt
import corner
import jax
import jax.numpy as jnp
from jaxtyping import Float

from jimgw.prior import UniformPrior, CombinePrior
from jimgw.single_event.transforms import ChirpMassMassRatioToComponentMassesTransform
from jimgw.base import LikelihoodBase
from jimgw.jim import Jim

params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)

# Likelihood for this test:

class MyLikelihood(LikelihoodBase):
    """Simple toy likelihood: Gaussian centered on the true component masses"""
    
    true_m1: Float
    true_m2: Float
    
    def __init__(self,
                 true_m1: Float,
                 true_m2: Float):
        
        self.true_m1 = true_m1
        self.true_m2 = true_m2
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        m1, m2 = params['m_1'], params['m_2']
        m1_std = 0.1
        m2_std = 0.1
        return -0.5 * (((m1 - self.true_m1) / m1_std)**2 + ((m2 - self.true_m2) / m2_std)**2)

# Setup
true_m1 = 1.6
true_m2 = 1.4
true_mc = (true_m1 * true_m2)**(3/5) / (true_m1 + true_m2)**(1/5)
true_q = true_m2 / true_m1

# Priors
eps = 0.5 # half of width of the chirp mass prior
mc_prior = UniformPrior(true_mc - eps, true_mc + eps, parameter_names=['M_c'])
q_prior = UniformPrior(0.125, 1.0, parameter_names=['q'])
combine_prior = CombinePrior([mc_prior, q_prior])

# Likelihood and transform
likelihood = MyLikelihood(true_m1, true_m2)
mass_transform = ChirpMassMassRatioToComponentMassesTransform

print("Checking mass_transform repr")
print(repr(mass_transform))

# Other stuff we have to give to Jim to make it work
step = 5e-3
local_sampler_arg = {"step_size": step * jnp.eye(2)}

# Jim:
jim = Jim(likelihood, 
          combine_prior, 
          likelihood_transforms=[mass_transform],
          n_chains = 10,
          parameter_names=['M_c', 'q'],
          n_loop_training=2,
          n_loop_production=2,
          local_sampler_arg=local_sampler_arg)

jim.sample(jax.random.PRNGKey(0))
jim.print_summary()
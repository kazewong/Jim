
import jax
import jax.numpy as jnp

from jimgw.single_event.runManager import (SingleEventPERunManager,
                                           SingleEventRun)

jax.config.update("jax_enable_x64", True)

mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[5, 5].set(1e-3)
mass_matrix = mass_matrix * 3e-3
local_sampler_arg = {"step_size": mass_matrix}
bounds = jnp.array(
    [
        [10.0, 40.0],
        [0.125, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [0.0, 2000.0],
        [-0.05, 0.05],
        [0.0, 2 * jnp.pi],
        [-1.0, 1.0],
        [0.0, jnp.pi],
        [0.0, 2 * jnp.pi],
        [-1.0, 1.0],
    ]
)

run = SingleEventRun(
    seed=0,
    detectors=["H1", "L1"],
    priors={
        "M_c": {"name": "Unconstrained_Uniform", "xmin": 10.0, "xmax": 80.0},
        "q": {"name": "MassRatio"},
        "s1_z": {"name": "Unconstrained_Uniform", "xmin": -1.0, "xmax": 1.0},
        "s2_z": {"name": "Unconstrained_Uniform", "xmin": -1.0, "xmax": 1.0},
        "d_L": {"name": "Unconstrained_Uniform", "xmin": 0.0, "xmax": 2000.0},
        "t_c": {"name": "Unconstrained_Uniform", "xmin": -0.05, "xmax": 0.05},
        "phase_c": {"name": "Unconstrained_Uniform", "xmin": 0.0, "xmax": 2 * jnp.pi},
        "cos_iota": {"name": "CosIota"},
        "psi": {"name": "Unconstrained_Uniform", "xmin": 0.0, "xmax": jnp.pi},
        "ra": {"name": "Unconstrained_Uniform", "xmin": 0.0, "xmax": 2 * jnp.pi},
        "sin_dec": {"name": "SinDec"},
    },
    waveform_parameters={"name": "RippleIMRPhenomD", "f_ref": 20.0},
    jim_parameters={
        "n_loop_training": 10,
        "n_loop_production": 10,
        "n_local_steps": 150,
        "n_global_steps": 150,
        "n_chains": 500,
        "n_epochs": 50,
        "learning_rate": 0.001,
        "n_max_examples": 45000,
        "momentum": 0.9,
        "batch_size": 50000,
        "use_global": True,
        "keep_quantile": 0.0,
        "train_thinning": 1,
        "output_thinning": 10,
        "local_sampler_arg": local_sampler_arg,
    },
    likelihood_parameters={"name": "HeterodynedTransientLikelihoodFD", "bounds": bounds},
    injection=True,
    injection_parameters={
        "M_c": 28.6,
        "eta": 0.24,
        "s1_z": 0.05,
        "s2_z": 0.05,
        "d_L": 440.0,
        "t_c": 0.0,
        "phase_c": 0.0,
        "iota": 0.5,
        "psi": 0.7,
        "ra": 1.2,
        "dec": 0.3,
    },
    data_parameters={
        "trigger_time": 1126259462.4,
        "duration": 4,
        "post_trigger_duration": 2,
        "f_min": 20.0,
        "f_max": 1024.0,
        "tukey_alpha": 0.2,
        "f_sampling": 4096.0,
    },
)

run_manager = SingleEventPERunManager(run=run)
run_manager.jim.sample(jax.random.PRNGKey(42))

# plot the corner plot and diagnostic plot
run_manager.plot_corner()
run_manager.plot_diagnostic()


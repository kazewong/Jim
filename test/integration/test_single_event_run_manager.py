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

run = SingleEventRun(
    seed=0,
    detectors=["H1", "L1"],
    data_parameters={
        "trigger_time": 1126259462.4,
        "duration": 4,
        "post_trigger_duration": 2,
        "f_min": 20.0,
        "f_max": 1024.0,
        "tukey_alpha": 0.2,
        "f_sampling": 4096.0,
    },
    priors={
        "M_c": {"name": "UniformPrior", "xmin": 10.0, "xmax": 80.0},
        "q": {"name": "UniformPrior", "xmin": 0.0, "xmax": 1.0},
        "s1_z": {"name": "UniformPrior", "xmin": -1.0, "xmax": 1.0},
        "s2_z": {"name": "UniformPrior", "xmin": -1.0, "xmax": 1.0},
        "d_L": {"name": "UniformPrior", "xmin": 1.0, "xmax": 2000.0},
        "t_c": {"name": "UniformPrior", "xmin": -0.05, "xmax": 0.05},
        "phase_c": {"name": "UniformPrior", "xmin": 0.0, "xmax": 2 * jnp.pi},
        "iota": {"name": "SinePrior"},
        "psi": {"name": "UniformPrior", "xmin": 0.0, "xmax": jnp.pi},
        "ra": {"name": "UniformPrior", "xmin": 0.0, "xmax": 2 * jnp.pi},
        "dec": {"name": "CosinePrior"},
    },
    waveform_parameters={"name": "RippleIMRPhenomD", "f_ref": 20.0},
    likelihood_parameters={"name": "TransientLikelihoodFD"},
    sample_transforms=[
        {"name": "BoundToUnbound", "name_mapping": [["M_c"], ["M_c_unbounded"]], "original_lower_bound": 10.0, "original_upper_bound": 80.0,},
        {"name": "BoundToUnbound", "name_mapping": [["q"], ["q_unbounded"]], "original_lower_bound": 0.0, "original_upper_bound": 1.0,},
        {"name": "BoundToUnbound", "name_mapping": [["s1_z"], ["s1_z_unbounded"]], "original_lower_bound": -1.0, "original_upper_bound": 1.0,},
        {"name": "BoundToUnbound", "name_mapping": [["s2_z"], ["s2_z_unbounded"]], "original_lower_bound": -1.0, "original_upper_bound": 1.0,},
        {"name": "BoundToUnbound", "name_mapping": [["d_L"], ["d_L_unbounded"]], "original_lower_bound": 1.0, "original_upper_bound": 2000.0,},
        {"name": "BoundToUnbound", "name_mapping": [["t_c"], ["t_c_unbounded"]], "original_lower_bound": -0.05, "original_upper_bound": 0.05,},
        {"name": "BoundToUnbound", "name_mapping": [["phase_c"], ["phase_c_unbounded"]], "original_lower_bound": 0.0, "original_upper_bound": 2 * jnp.pi,},
        {"name": "BoundToUnbound", "name_mapping": [["iota"], ["iota_unbounded"]], "original_lower_bound": 0.0, "original_upper_bound": jnp.pi,},
        {"name": "BoundToUnbound", "name_mapping": [["psi"], ["psi_unbounded"]], "original_lower_bound": 0.0, "original_upper_bound": jnp.pi,},
        {"name": "BoundToUnbound", "name_mapping": [["ra"], ["ra_unbounded"]], "original_lower_bound": 0.0, "original_upper_bound": 2 * jnp.pi,},
        {"name": "BoundToUnbound", "name_mapping": [["dec"], ["dec_unbounded"]], "original_lower_bound": 0.0, "original_upper_bound": jnp.pi,},
    ],
    likelihood_transforms=[
        {"name": "MassRatioToSymmetricMassRatioTransform", "name_mapping": [["q"], ["eta"]]},
    ],
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
    jim_parameters={
        "n_loop_training": 1,
        "n_loop_production": 1,
        "n_local_steps": 5,
        "n_global_steps": 5,
        "n_chains": 4,
        "n_epochs": 2,
        "learning_rate": 1e-4,
        "n_max_examples": 30,
        "momentum": 0.9,
        "batch_size": 100,
        "use_global": True,
        "train_thinning": 1,
        "output_thinning": 1,
        "local_sampler_arg": local_sampler_arg,
    },
)

run_manager = SingleEventPERunManager(run=run)
run_manager.sample()

# plot the corner plot and diagnostic plot
run_manager.plot_corner()
run_manager.plot_diagnostic()
run_manager.save_summary()

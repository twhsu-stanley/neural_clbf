from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.systems import InvertedPendulumCartSINDy
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
)
from neural_clbf.training.utils import current_git_hash

import pysindy as ps

torch.multiprocessing.set_sharing_strategy("file_system")

controller_period = 0.01
simulation_dt = 0.01

start_x = torch.tensor(
    [
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, -0.5, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, -0.5, 0.0, 0.0],
    ]
)

def main(args):
    # Define the scenarios
    nominal_params = {"M": 1.0, "m": 1.0, "L": 0.5, "Kd": 10.0}
    scenarios = [
        nominal_params,
    ]

    # Define the dynamics model
    dynamics_model = InvertedPendulumCartSINDy(
        nominal_params,
        dt = simulation_dt,
        controller_dt = controller_period,
        scenarios = scenarios
    )

    # Initialize the DataModule
    initial_conditions = [
        (-8.0, 8.0), # z
        (-10.0, 10.0), # z_dot
        (-1.2, 1.2), # theta
        (-10.0, 10.0), # theta_dot
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode = 0,
        trajectory_length = 1,
        fixed_samples = 10000,
        max_points = 100000,
        val_split = 0.1,
        batch_size = 64,
        # quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.4},
    )

    # Define the experiment suite
    V_contour_experiment_1 = CLFContourExperiment(
        "V_Contour",
        domain = [(-10.0, 10.0), (-10.0, 10.0)],
        n_grid = 30,
        x_axis_index = InvertedPendulumCartSINDy.Z,
        y_axis_index = InvertedPendulumCartSINDy.Z_DOT,
        x_axis_label = "z",
        y_axis_label = "$\\dot{z}$",
        plot_unsafe_region = False,
    )
    V_contour_experiment_2 = CLFContourExperiment(
        "V_Contour",
        domain = [(-np.pi, np.pi), (-10.0, 10.0)],
        n_grid = 30,
        x_axis_index = InvertedPendulumCartSINDy.THETA,
        y_axis_index = InvertedPendulumCartSINDy.THETA_DOT,
        x_axis_label = "$\\theta$",
        y_axis_label = "$\\dot{\\theta}$",
        plot_unsafe_region = False,
    )

    rollout_experiment_1 = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        InvertedPendulumCartSINDy.Z,
        "z",
        InvertedPendulumCartSINDy.Z_DOT,
        "$\\dot{z}$",
        scenarios = scenarios,
        n_sims_per_start = 1,
        t_sim = 2.0,
    )
    rollout_experiment_2 = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        InvertedPendulumCartSINDy.THETA,
        "$\\theta$",
        InvertedPendulumCartSINDy.THETA_DOT,
        "$\\dot{\\theta}$",
        scenarios = scenarios,
        n_sims_per_start = 1,
        t_sim = 2.0,
    )
    experiment_suite = ExperimentSuite(
        []
    )
    
    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite = experiment_suite,
        clbf_hidden_layers = 4,
        clbf_hidden_size = 64 * 2,
        clf_lambda = 0.0,
        safe_level = 1.0,
        controller_period = controller_period,
        clf_relaxation_penalty = 1e6,
        primal_learning_rate = 1e-3,
        num_init_epochs = 11,
        epochs_per_episode = 100,
        barrier = False,
        disable_gurobi = True,
        #add_nominal = True,
        #normalize_V_nominal = True,
        roa_regulator = False,
        #roa_regulator_alpha = 6.0,
        cp_learning = True,
        solver_args = {"max_iters": 1000}
    )
    #solver_args = {"solve_method": "ECOS", "max_iters": 1000, "reltol": 1e-8}
    #solver_args = {"eps": 1e-8, "max_iters": 10000, "acceleration_lookback": 0}

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/inverted_pendulum_cart_sindy",
        name = f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger = tb_logger,
        reload_dataloaders_every_epoch = True,
        gradient_clip_val = 0.5,
        max_epochs = 121,
        stochastic_weight_avg = True
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)

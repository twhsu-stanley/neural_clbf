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
from neural_clbf.systems import DoubleInvertedPendulum
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
)
from neural_clbf.training.utils import current_git_hash


torch.multiprocessing.set_sharing_strategy("file_system")

controller_period = 0.001

start_x = torch.tensor(
    [
        [0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
    ]
)
simulation_dt = 0.001


def main(args):
    # Define the scenarios
    nominal_params = {"M": 5.0, "m1": 2.0, "m2": 1.5, "L1": 0.5, "L2": 0.25}
    scenarios = [
        nominal_params,
    ]

    # Define the dynamics model
    dynamics_model = DoubleInvertedPendulum(
        nominal_params,
        dt = simulation_dt,
        controller_dt = controller_period,
        scenarios = scenarios,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-10.0, 10.0), # x
        (-20.0, 20.0), # x_dot
        (-np.pi, np.pi), # theta_1
        (-np.pi*15, np.pi*15), # theta_1_dot
        (-np.pi, np.pi), # theta_2
        (-np.pi*15, np.pi*15), # theta_2_dot
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
        domain = [(-np.pi, np.pi), (-np.pi*15, np.pi*15)],
        n_grid = 30,
        x_axis_index = DoubleInvertedPendulum.THETA_1,
        y_axis_index = DoubleInvertedPendulum.THETA_1_DOT,
        x_axis_label = "$\\theta_1$",
        y_axis_label = "$\\dot{\\theta}_1$",
        plot_unsafe_region = False,
    )
    V_contour_experiment_2 = CLFContourExperiment(
        "V_Contour",
        domain = [(-np.pi, np.pi), (-np.pi*15, np.pi*15)],
        n_grid = 30,
        x_axis_index = DoubleInvertedPendulum.THETA_2,
        y_axis_index = DoubleInvertedPendulum.THETA_2_DOT,
        x_axis_label = "$\\theta_2$",
        y_axis_label = "$\\dot{\\theta}_2$",
        plot_unsafe_region = False,
    )
    V_contour_experiment_3 = CLFContourExperiment(
        "V_Contour",
        domain = [(-10.0, 10.0), (-20.0, 20.0)],
        n_grid = 30,
        x_axis_index = DoubleInvertedPendulum.X,
        y_axis_index = DoubleInvertedPendulum.X_DOT,
        x_axis_label = "x",
        y_axis_label = "$\\dot{x}$",
        plot_unsafe_region = False,
    )

    rollout_experiment_1 = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        DoubleInvertedPendulum.THETA_1,
        "$\\theta_1$",
        DoubleInvertedPendulum.THETA_1_DOT,
        "$\\dot{\\theta}_1$",
        scenarios = scenarios,
        n_sims_per_start = 1,
        t_sim = 3.0,
    )
    rollout_experiment_2 = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        DoubleInvertedPendulum.THETA_2,
        "$\\theta_2$",
        DoubleInvertedPendulum.THETA_2_DOT,
        "$\\dot{\\theta}_2$",
        scenarios = scenarios,
        n_sims_per_start = 1,
        t_sim = 3.0,
    )
    rollout_experiment_3 = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        DoubleInvertedPendulum.X,
        "x",
        DoubleInvertedPendulum.X_DOT,
        "$\\dot{x}$",
        scenarios = scenarios,
        n_sims_per_start = 1,
        t_sim = 3.0,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment_1, V_contour_experiment_2, V_contour_experiment_3, 
                                        rollout_experiment_1, rollout_experiment_2, rollout_experiment_3])
    
    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite = experiment_suite,
        clbf_hidden_layers = 3,
        clbf_hidden_size = 64,
        clf_lambda = 0.5,
        safe_level = 0.1,
        controller_period = controller_period,
        clf_relaxation_penalty = 1e3,
        primal_learning_rate = 1e-3,
        num_init_epochs = 5,
        epochs_per_episode = 100,
        barrier = False,
        disable_gurobi = True,
        #add_nominal = True,
        #normalize_V_nominal = True,
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/double_inverted_pendulum",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger = tb_logger,
        reload_dataloaders_every_epoch = True,
        gradient_clip_val = 0.5,
        max_epochs = 151,
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

from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from neural_clbf.controllers import NeuralCBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.systems import DubinsCar
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
)
from neural_clbf.training.utils import current_git_hash


torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 128
controller_period = 0.01

start_x = torch.tensor(
    [
        [-5, 0.0, 0.0],
    ]
)
simulation_dt = 0.01


def main(args):
    # Define the scenarios
    nominal_params = {
        "v": 0.5,
    }
    scenarios = [
        nominal_params,
    ]

    # Define the dynamics model
    dynamics_model = DubinsCar(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-1.0, 1.0),  # x
        (-1.0, 1.0),  # y
        (-0.01, 0.01),  # theta
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=30000,
        max_points=50000,
        val_split=0.1,
        batch_size=batch_size,
        #quotas={"safe": 0.4, "unsafe": 0.2},
    )

    # Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-7.0, 7.0), (-7.0, 7.0)],
        n_grid=25,
        x_axis_index=DubinsCar.X,
        y_axis_index=DubinsCar.Y,
        x_axis_label="$x$",
        y_axis_label="$y$",
    )
    rollout_state_space_experiment = RolloutStateSpaceExperiment(
        "Rollout State Space",
        start_x,
        plot_x_index=DubinsCar.X,
        plot_x_label="$x$",
        plot_y_index=DubinsCar.Y,
        plot_y_label="$y$",
        scenarios=[nominal_params],
        n_sims_per_start=1,
        t_sim=5.0,
    )
    experiment_suite = ExperimentSuite(
        [
            V_contour_experiment,
            rollout_state_space_experiment,
        ]
    )

    # Initialize the controller
    clbf_controller = NeuralCBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        cbf_hidden_layers=4,
        cbf_hidden_size=128,
        cbf_lambda=0.1,
        controller_period=controller_period,
        cbf_relaxation_penalty=1e4,
        scale_parameter=10.0,
        primal_learning_rate=5e-4,
        learn_shape_epochs=60,
        #use_relu=True,
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/dubins_car",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, reload_dataloaders_every_epoch=True, gradient_clip_val = 0.5, max_epochs=301
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)

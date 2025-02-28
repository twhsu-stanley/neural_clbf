from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from neural_clbf.controllers import NeuralCBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.systems import DubinsCarSINDy
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
        [-4, 0.0, 0.0],
    ]
)
simulation_dt = 0.01


def main(args):
    # Define the scenarios
    nominal_params = {
        "v": 1.0,
    }
    scenarios = [
        nominal_params,
    ]

    # Define the dynamics model
    dynamics_model = DubinsCarSINDy(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-5.0, 5.0),  # x
        (-5.0, 5.0),  # y
        (-0.1, 0.1),  # theta
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=10000,
        max_points=20000,
        val_split=0.1,
        batch_size=batch_size,
        #quotas={"safe": 0.5},
    )

    # Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-5.0, 5.0), (-5.0, 5.0)],
        n_grid=10,
        x_axis_index=DubinsCarSINDy.X,
        y_axis_index=DubinsCarSINDy.Y,
        x_axis_label="$x$",
        y_axis_label="$y$",
    )
    rollout_state_space_experiment = RolloutStateSpaceExperiment(
        "Rollout State Space",
        start_x,
        plot_x_index=DubinsCarSINDy.X,
        plot_x_label="$x$",
        plot_y_index=DubinsCarSINDy.Y,
        plot_y_label="$y$",
        scenarios=[nominal_params],
        n_sims_per_start=1,
        t_sim=1.0,
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
        cbf_hidden_size=256,
        cbf_lambda=1.0,
        controller_period=controller_period,
        cbf_relaxation_penalty=1e4,
        scale_parameter=1.0,
        primal_learning_rate=5e-3,
        learn_shape_epochs=50,
        cp_learning = True,
        use_relu=True,
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/dubins_car_sindy",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, reload_dataloaders_every_epoch=True, max_epochs=301 #gradient_clip_val = 0.5,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)

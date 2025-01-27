import torch
import matplotlib
from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.systems import CartPoleSINDy
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
)
import numpy as np

matplotlib.use('TkAgg')

def plot_inverted_pendulum_cart():
    # Load the checkpoint file. This should include the experiment suite used during training.
    log_file = "./logs/cartpole_sindy/commit_77ba6fc/version_0/checkpoints/epoch=120-step=19880.ckpt"
    
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Define the scenarios
    nominal_params = {"M": 1.0, "m": 0.3, "L": 1.0, "b": 0.0}
    scenarios = [
        nominal_params,
    ]

    # Update parameters
    start_x = torch.tensor(
        [
            #[0.0, 0.0, 0.0, 8.0],
            #[0.0, 0.0, 0.0, -8.0],
            #[0.0, 0.0, 0.0, 5.0],
            #[0.0, 0.0, 0.0, -5.0],
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.3, 0.0, 0.0],
            #[0.0, 0.0, 0.9, 0.0],
            #[0.0, 0.0, -0.9, 0.0],
            #[0.0, 0.0, 1.0, 0.0],
            #[0.0, 0.0, -1.0, 0.0],
            #[0.0, 0.0, 1.1, -0.2],
            #[0.0, 0.0, -1.1, 0.2],
            #[0.0, 0.0, 1.0, 1.0],
            #[0.0, 0.0, 1.0, -1.0],
            #[0.0, 0.0, -1.0, 1.0],
            #[0.0, 0.0, -1.0, -1.0],
            #[0.0, 1.0, 0.0, 0.0],
            #[0.0, -1.0, 0.0, 0.0],
        ]
    )

    # Define the experiment suite
    V_contour_experiment_1 = CLFContourExperiment(
        "V_Contour",
        domain = [(-1.0, 1.0), (-1.5, 1.5)],
        n_grid = 30,
        x_axis_index = CartPoleSINDy.Z,
        y_axis_index = CartPoleSINDy.V,
        x_axis_label = "z",
        y_axis_label = "$\\dot{z}$",
        plot_unsafe_region = False,
    )
    V_contour_experiment_2 = CLFContourExperiment(
        "V_Contour",
        domain = [(-np.pi/6, np.pi/6), (-1.0, 1.0)],
        n_grid = 30,
        x_axis_index = CartPoleSINDy.THETA,
        y_axis_index = CartPoleSINDy.OMEGA,
        x_axis_label = "$\\theta$",
        y_axis_label = "$\\dot{\\theta}$",
        plot_unsafe_region = False,
    )

    rollout_experiment_1 = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        CartPoleSINDy.Z,
        "z",
        CartPoleSINDy.V,
        "$\\dot{z}$",
        scenarios = scenarios,
        n_sims_per_start = 1,
        t_sim = 2.0,
    )
    rollout_experiment_2 = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        CartPoleSINDy.THETA,
        "$\\theta$",
        CartPoleSINDy.OMEGA,
        "$\\dot{\\theta}$",
        scenarios = scenarios,
        n_sims_per_start = 1,
        t_sim = 2.0,
    )
    #experiment_suite = ExperimentSuite([V_contour_experiment_1, V_contour_experiment_2, rollout_experiment_2])
    experiment_suite = ExperimentSuite([rollout_experiment_2])
    
    neural_controller.experiment_suite = experiment_suite

    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_plot(
        neural_controller, display_plots=True
    )

if __name__ == "__main__":
    
    plot_inverted_pendulum_cart()

import torch
import matplotlib
from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.systems import InvertedPendulumCart
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
)
import numpy as np

matplotlib.use('TkAgg')

def plot_inverted_pendulum_cart():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_file = "./logs/inverted_pendulum_cart/commit_ca448fb/version_5/checkpoints/epoch=120-step=19880.ckpt" # nominal model
    #log_file = "./logs/inverted_pendulum_cart_sindy/commit_ca448fb/version_31/checkpoints/epoch=120-step=19880.ckpt" # SINDy model
    
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Define the scenarios
    nominal_params = {"M": 1.0, "m": 1.0, "L": 0.5, "Kd": 10.0}
    scenarios = [
        nominal_params,
    ]

    # Update parameters
    start_x = torch.tensor(
        [
            [0.0, 0.0, 0.0, 8.0],
            [0.0, 0.0, 0.0, -8.0],
            [0.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, -5.0],
            [0.0, 0.0, 1.1, 0.0],
            [0.0, 0.0, -1.1, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            #[0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, -1.0, 1.0],
            #[0.0, 0.0, -1.0, -1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ]
    )

    # Define the experiment suite
    V_contour_experiment_1 = CLFContourExperiment(
        "V_Contour",
        domain = [(-10.0, 10.0), (-20.0, 20.0)],
        n_grid = 30,
        x_axis_index = InvertedPendulumCart.Z,
        y_axis_index = InvertedPendulumCart.Z_DOT,
        x_axis_label = "z",
        y_axis_label = "$\\dot{z}$",
        plot_unsafe_region = False,
    )
    V_contour_experiment_2 = CLFContourExperiment(
        "V_Contour",
        domain = [(-np.pi*2, np.pi*2), (-np.pi*10, np.pi*10)],
        n_grid = 30,
        x_axis_index = InvertedPendulumCart.THETA,
        y_axis_index = InvertedPendulumCart.THETA_DOT,
        x_axis_label = "$\\theta$",
        y_axis_label = "$\\dot{\\theta}$",
        plot_unsafe_region = False,
    )

    rollout_experiment_1 = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        InvertedPendulumCart.Z,
        "z",
        InvertedPendulumCart.Z_DOT,
        "$\\dot{z}$",
        scenarios = scenarios,
        n_sims_per_start = 1,
        t_sim = 2.0,
    )
    rollout_experiment_2 = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        InvertedPendulumCart.THETA,
        "$\\theta$",
        InvertedPendulumCart.THETA_DOT,
        "$\\dot{\\theta}$",
        scenarios = scenarios,
        n_sims_per_start = 1,
        t_sim = 5.0,
    )
    #experiment_suite = ExperimentSuite([V_contour_experiment_1, V_contour_experiment_2, rollout_experiment_2, rollout_experiment_1])
    experiment_suite = ExperimentSuite([rollout_experiment_2])
    
    neural_controller.experiment_suite = experiment_suite

    #neural_controller.experiment_suite.experiments[2].start_x = start_x
    #neural_controller.experiment_suite.experiments[3].start_x = start_x

    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_plot(
        neural_controller, display_plots=True
    )

if __name__ == "__main__":
    # eval_inverted_pendulum()
    plot_inverted_pendulum_cart()

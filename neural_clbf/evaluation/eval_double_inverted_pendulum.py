import torch
import matplotlib
from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.systems import DoubleInvertedPendulum
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
)
import numpy as np

matplotlib.use('TkAgg')

def plot_double_inverted_pendulum():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    #log_file = "logs/double_inverted_pendulum/commit_ca448fb/version_27/checkpoints/epoch=100-step=14240.ckpt" # low loss
    log_file = "logs/double_inverted_pendulum/commit_ca448fb/version_35/checkpoints/epoch=150-step=28340.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Define the scenarios
    nominal_params = {"M": 5.0, "m1": 2.0, "m2": 1.5, "L1": 0.5, "L2": 0.25}
    scenarios = [
        nominal_params,
    ]

    # Update parameters
    start_x = torch.tensor(
        [   [0.0, 0.0, np.pi/2, 0.1, np.pi/4, 0.1],
            [0.0, 0.0, -0.1, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1, 0.0, -0.1, 0.0],
            [0.0, 0.0, -0.1, 0.0, -0.1, 0.0],
            #[0.1, 0.1, -0.1, 0.0, -0.1, 0.0],
            #[0.1, 0.1, -0.1, 0.0, -0.1, 0.0],
        ]
    )
    # Define the experiment suite
    V_contour_experiment_1 = CLFContourExperiment(
        "V_Contour",
        domain = [(-np.pi*2, np.pi*2), (-np.pi*2, np.pi*2)],
        n_grid = 30,
        x_axis_index = DoubleInvertedPendulum.THETA_1,
        y_axis_index = DoubleInvertedPendulum.THETA_2,
        x_axis_label = "$\\theta_1$",
        y_axis_label = "$\\dot{\\theta}_1$",
        plot_unsafe_region = False,
    )
    V_contour_experiment_2 = CLFContourExperiment(
        "V_Contour",
        domain = [(-np.pi*2, np.pi*2), (-np.pi*10, np.pi*10)],
        n_grid = 30,
        x_axis_index = DoubleInvertedPendulum.THETA_2,
        y_axis_index = DoubleInvertedPendulum.THETA_2_DOT,
        x_axis_label = "$\\theta_2$",
        y_axis_label = "$\\dot{\\theta}_2$",
        plot_unsafe_region = False,
    )
    V_contour_experiment_3 = CLFContourExperiment(
        "V_Contour",
        domain = [(-50.0, 50.0), (-20.0, 20.0)],
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
    neural_controller.experiment_suite = experiment_suite
    #neural_controller.experiment_suite.experiments[3].start_x = start_x
    #neural_controller.experiment_suite.experiments[4].start_x = start_x
    #neural_controller.experiment_suite.experiments[5].start_x = start_x

    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_plot(
        neural_controller, display_plots=True
    )


if __name__ == "__main__":
    # eval_inverted_pendulum()
    plot_double_inverted_pendulum()

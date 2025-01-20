import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

from neural_clbf.controllers import NeuralCBFController


matplotlib.use('TkAgg')


def plot_linear_satellite():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_file = "logs/dubins_car/commit_2b14ba7/version_0/checkpoints/epoch=200-step=35375.ckpt"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_file)

    # Tweak parameters
    neural_controller.cbf_relaxation_penalty = 1e2
    neural_controller.clf_lambda = 0.1
    neural_controller.controller_period = 0.01

    # Tweak experiments
    neural_controller.experiment_suite.experiments[0].n_grid = 50  # 200
    neural_controller.experiment_suite.experiments[1].t_sim = 5.0
    neural_controller.experiment_suite.experiments[1].start_x = torch.tensor(
        [[-5, 0.0, 0.0]]
    )
    neural_controller.experiment_suite.experiments[1].other_index = [2]
    neural_controller.experiment_suite.experiments[1].other_label = ["$z$"]

    # Run the experiments and save the results
    #grid_df = neural_controller.experiment_suite.experiments[0].run(neural_controller)
    neural_controller.experiment_suite.run_all_and_plot(
        neural_controller, display_plots=True
    )
    traj_df = neural_controller.experiment_suite.experiments[1].run(neural_controller)

    fig, ax = plt.subplots(1, 1)
    ax.plot(traj_df["$x$"], traj_df["$y$"])
    circle1 = plt.Circle((0, 0), 1.0, color='r')
    ax.add_patch(circle1)
    ax.plot()
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.grid(True)
    #ax.set_title("CBF")
    plt.show()


if __name__ == "__main__":
    plot_linear_satellite()

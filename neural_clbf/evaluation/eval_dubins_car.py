import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

from neural_clbf.controllers import NeuralCBFController
from neural_clbf.conformal_prediction.clf_cp_sim_utils import clf_simulation, create_clf_qp_cp_cvxpylayers_solver

matplotlib.use('TkAgg')


def plot_linear_satellite():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_file = "logs/dubins_car/commit_2b14ba7/version_0/checkpoints/epoch=200-step=35375.ckpt"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_file)

    # Create a QP solver
    clf_qp_solver = create_clf_qp_cp_cvxpylayers_solver(neural_controller)

    # Set up initial conditions for the sim
    start_x = torch.tensor(
        [[-3, 0.0, 0.0], [-4, 0, 0], [-2.5, 0.0, 1.0],[-3,-0.5,0]]
    )

    T = 10.0
    delta_t = neural_controller.dynamics_model.dt
    num_timesteps = int(T // delta_t)
    
    solver_args = {"eps": 1e-8}
    u_history, r_history, x_history, V_history, p_history = \
        clf_simulation(neural_controller, clf_qp_solver, start_x, T = T, solver_args = solver_args, plot = False)
    
    fig, ax = plt.subplots(1, 1)
    ax.plot((x_history[:,0,:]).squeeze().T, (x_history[:,1,:]).squeeze().T)
    #(x_history[:,d,:]).squeeze().T
    # Plot the unsafe region
    circle1 = plt.Circle((0, 0), 1.0, color='r')
    ax.add_patch(circle1)
    ax.plot()
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.grid(True)
    #ax.set_title("CBF")

    fig, ax = plt.subplots(1, 1)
    for i in range(start_x.shape[0]):
        ax.plot(np.arange(num_timesteps) * delta_t, V_history[i,:])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("V(x)")
    ax.grid(True)
    ax.set_title("State Norms and CBFs")

    fig, ax = plt.subplots(1, 1)
    for i in range(start_x.shape[0]):
        ax.plot(np.arange(num_timesteps) * delta_t, p_history[i,:])
    ax.set_ylabel("p = Vdot + lambda*V")
    ax.set_xlabel("Time (s)")
    ax.grid(True)
    ax.set_title("CBF Constraints")

    plt.show()

    # Tweak parameters
    #neural_controller.cbf_relaxation_penalty = 1e2
    #neural_controller.clf_lambda = 0.1
    #neural_controller.controller_period = 0.01

    # Tweak experiments
    """
    neural_controller.experiment_suite.experiments[0].n_grid = 10  # 200
    neural_controller.experiment_suite.experiments[1].t_sim = 10.0
    neural_controller.experiment_suite.experiments[1].start_x = torch.tensor(
        [[-3, 0.0, 0.0]]
    )
    neural_controller.experiment_suite.experiments[1].other_index = [2]
    neural_controller.experiment_suite.experiments[1].other_label = ["$z$"]

    # Run the experiments and save the results
    #grid_df = neural_controller.experiment_suite.experiments[0].run(neural_controller)
    neural_controller.experiment_suite.run_all_and_plot(neural_controller, display_plots=True)
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
    """


if __name__ == "__main__":
    plot_linear_satellite()

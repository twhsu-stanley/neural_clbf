import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
)
from neural_clbf.controllers import NeuralCBFController
from neural_clbf.simulation.clf_sim_utils import clf_simulation_gurobi

matplotlib.use('TkAgg')

def plot_dubins_cbf():
    # Load the checkpoint file. This should include the experiment suite used during training
    # Circular obstacle worked
    # CP
    log_file = "logs/dubins_car_sindy/commit_c879c51/version_0/checkpoints/epoch=100-step=7170.ckpt"
    # Non CP
    log_file = "logs/dubins_car_sindy/commit_84ef882/version_0/checkpoints/epoch=100-step=7170.ckpt"
    
    neural_controller = NeuralCBFController.load_from_checkpoint(log_file)

    # Tweak parameters
    #neural_controller.cbf_relaxation_penalty = 1e9
    #neural_controller.clf_lambda = 0.1
    #neural_controller.controller_period = 0.001
    #neural_controller.disable_gurobi = True

    V_contour_experiment_1 = CLFContourExperiment(
        "V_Contour",
        domain = [(-6.0, 6.0), (-6.0, 6.0)],
        n_grid = 40,
        x_axis_index = 0,
        y_axis_index = 1,
        x_axis_label = "p_x",
        y_axis_label = "p_y",
    )
    V_contour_experiment_2 = CLFContourExperiment(
        "V_Contour",
        domain = [(-6.0, 6.0), (-100/180*np.pi, 100/180*np.pi)],
        n_grid = 40,
        x_axis_index = 0,
        y_axis_index = 2,
        x_axis_label = "p_x",
        y_axis_label = "theta",
    )
    V_contour_experiment_3 = CLFContourExperiment(
        "V_Contour",
        domain = [(-6.0, 6.0), (-100/180*np.pi, 100/180*np.pi)],
        n_grid = 40,
        x_axis_index = 1,
        y_axis_index = 2,
        x_axis_label = "p_y",
        y_axis_label = "theta",
    )
    neural_controller.experiment_suite = ExperimentSuite([V_contour_experiment_1])
    #neural_controller.experiment_suite.run_all_and_plot(neural_controller, display_plots = True)

    # Set up initial conditions for the sim
    N = 10 # number of trajectories
    start_x = torch.hstack((
        torch.rand(N,1) * 1 - 4,
        torch.rand(N,1) * 1 - 0.5,
        torch.rand(N,1) * 0/180*np.pi - 0/180*np.pi
    ))
    #start_x = start_x[torch.pow(start_x[:,0] - 0.0, 2) + torch.pow(start_x[:,1] - 0.0, 2) >= 3**2, :]
    #N = start_x.shape[0] 

    T = 10
    delta_t = neural_controller.dynamics_model.dt
    num_timesteps = int(T // delta_t)
    
    u_history, x_history, V_history, p_history = clf_simulation_gurobi(neural_controller, start_x, T = T)
    
    fig, ax = plt.subplots(1, 1)
    ax.plot((x_history[:,0,:]).squeeze().T, (x_history[:,1,:]).squeeze().T)
    ax.add_patch(plt.Circle((0, 0), 2, alpha=0.5, facecolor="red"))
    #ax.add_patch(plt.Rectangle((-2, -2), 4, 8, alpha=0.5, facecolor="red"))
    ax.plot()
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.set_ylim(-4.0, 4.0)
    ax.set_xlim(-6.0, 6.0)
    ax.set_aspect('equal')
    ax.grid(True)

    fig, ax = plt.subplots(1, 1)
    for i in range(N):
        ax.plot(np.arange(num_timesteps) * delta_t, V_history[i,:])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("h(x)")
    ax.grid(True)
    ax.set_title("CBF")

    """
    fig, ax = plt.subplots(1, 1)
    for i in range(N):
        ax.plot(np.arange(num_timesteps) * delta_t, u_history[i,:].squeeze())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("u")
    ax.grid(True)
    ax.set_title("Control")

    fig, ax = plt.subplots(1, 1)
    for i in range(N):
        ax.plot(np.arange(num_timesteps) * delta_t, p_history[i,:])
    ax.set_ylabel("p = Vdot + lambda*h")
    ax.set_xlabel("Time (s)")
    ax.grid(True)
    ax.set_title("CBF Constraints")
    """

    plt.show()

if __name__ == "__main__":
    plot_dubins_cbf()

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
from neural_clbf.simulation.clf_sim_utils import clf_simulation_gurobi, create_clf_qp_cp_cvxpylayers_solver

matplotlib.use('TkAgg')

def plot_dubins_cbf():
    # Load the checkpoint file. This should include the experiment suite used during training
    log_file = "logs/dubins_car/commit_b3ccd6c/version_5/checkpoints/epoch=200-step=7235.ckpt"
    log_file = "logs/dubins_car/commit_622b8ae/version_0/checkpoints/epoch=173-step=6263.ckpt"
    
    neural_controller = NeuralCBFController.load_from_checkpoint(log_file)

    # Tweak parameters
    #neural_controller.cbf_relaxation_penalty = 1e4
    #neural_controller.clf_lambda = 0.5
    #neural_controller.controller_period = 0.01

    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain = [(2.0, 10.0), (1.0, 7.0)],
        n_grid = 40,
        x_axis_index = 0,
        y_axis_index = 1,
        x_axis_label = "p_x",
        y_axis_label = "p_y",
    )
    neural_controller.experiment_suite = ExperimentSuite([V_contour_experiment])
    neural_controller.experiment_suite.run_all_and_plot(
        neural_controller, display_plots = True
    )

    # Set up initial conditions for the sim
    N = 50 # number of trajectories
    start_x = torch.hstack((torch.rand(N,1) * 7 + 3, torch.rand(N,1) * 6 + 1, torch.rand(N,1) * 2 * np.pi - np.pi))
  
    start_x = start_x[torch.pow(start_x[:,0] - 5.0, 2) + torch.pow(start_x[:,1] - 4.0, 2) >= 3**2, :]
    N = start_x.shape[0]

    T = 8.0
    delta_t = neural_controller.dynamics_model.dt
    num_timesteps = int(T // delta_t)
    
    #solver_args = {"eps": 1e-8}
    u_history, x_history, V_history, p_history = clf_simulation_gurobi(neural_controller, start_x, T = T)
    
    fig, ax = plt.subplots(1, 1)
    ax.plot((x_history[:,0,:]).squeeze().T, (x_history[:,1,:]).squeeze().T)
    obstacle = plt.Circle((5, 4), 2, color='r')
    ax.add_patch(obstacle)
    ax.plot()
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.grid(True)

    fig, ax = plt.subplots(1, 1)
    for i in range(N):
        ax.plot(np.arange(num_timesteps) * delta_t, V_history[i,:])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("h(x)")
    ax.grid(True)
    ax.set_title("CBF")

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

    plt.show()

if __name__ == "__main__":
    plot_dubins_cbf()

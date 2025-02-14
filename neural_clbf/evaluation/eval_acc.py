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
    log_file = "logs/acc/commit_647f58e/version_1/checkpoints/epoch=250-step=70781.ckpt"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_file)

    #neural_controller.cbf_relaxation_penalty = 1e1
    #neural_controller.clf_lambda = 0.1

    # Create a QP solver
    clf_qp_solver = create_clf_qp_cp_cvxpylayers_solver(neural_controller)

    # Set up initial conditions for the sim
    start_x = torch.tensor(
        [[0.0, 25.0, 25.0], 
         [0.0, 28.0, 30.0], 
         [0.0, 25.0, 30.0],
         ]
    )

    T = 5.0
    delta_t = neural_controller.dynamics_model.dt
    num_timesteps = int(T // delta_t)
    
    #solver_args = {"eps": 1e-8}
    u_history, r_history, x_history, V_history, p_history = \
        clf_simulation(neural_controller, clf_qp_solver, start_x, T = T, plot = False)

    fig, ax = plt.subplots(1, 1)
    for i in range(start_x.shape[0]):
        ax.plot(np.arange(num_timesteps) * delta_t, (x_history[:,1,:]).squeeze().T)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("v (m/s)")
    ax.grid(True)

    fig, ax = plt.subplots(1, 1)
    for i in range(start_x.shape[0]):
        ax.plot(np.arange(num_timesteps) * delta_t, (x_history[:,2,:]).squeeze().T)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("z (m)")
    ax.grid(True)

    fig, ax = plt.subplots(1, 1)
    for i in range(start_x.shape[0]):
        ax.plot(np.arange(num_timesteps) * delta_t, V_history[i,:])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("V")
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

if __name__ == "__main__":
    plot_linear_satellite()

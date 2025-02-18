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
    # log_file = "saved_models/review/linear_satellite_cbf.ckpt"
    log_file = "logs/linear_satellite_cbf/relu/commit_e77aff5/version_2/checkpoints/epoch=198-step=35023.ckpt"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_file)

    # Tweak parameters
    neural_controller.cbf_relaxation_penalty = 1e9
    neural_controller.clf_lambda = 0.01
    #neural_controller.controller_period = 0.01

    #################################################
    # Create a QP solver
    clf_qp_solver = create_clf_qp_cp_cvxpylayers_solver(neural_controller)

    start_x = torch.tensor(
        [
            [0.5, 0.5, 0.0, -0.1, -0.1, -1.0],
            [0.3, 0.3, 0.1, -0.1, -0.3, -1.0],
            [0.3, 0.4, 0.2, -0.2, -0.4, -0.8],
            [0.3, 0.3, 0.3, -0.2, -0.3, -0.8],
        ]
    )

    T = 4.0
    delta_t = neural_controller.dynamics_model.dt
    num_timesteps = int(T // delta_t)
    
    #solver_args = {"eps": 1e-8}
    u_history, r_history, x_history, V_history, p_history = \
        clf_simulation(neural_controller, clf_qp_solver, start_x, T = T, plot = False)

    fig, ax = plt.subplots(1, 1)
    for i in range(start_x.shape[0]):
        ax.plot(np.arange(num_timesteps) * delta_t, np.linalg.norm((x_history[i,:3,:]), ord=2, axis=0))
    ax.hlines(0.25, 0, (num_timesteps-1)* delta_t, colors='r', linestyles='solid')
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance")
    ax.grid(True)

    fig, ax = plt.subplots(1, 1)
    for i in range(start_x.shape[0]):
        ax.plot(np.arange(num_timesteps) * delta_t, V_history[i,:])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("h")
    ax.grid(True)
    ax.set_title("CBF")

    fig, ax = plt.subplots(1, 1)
    for i in range(start_x.shape[0]):
        ax.plot(np.arange(num_timesteps) * delta_t, p_history[i,:])
    ax.set_ylabel("p = hdot + lambda*h")
    ax.set_xlabel("Time (s)")
    ax.grid(True)
    ax.set_title("CBF Constraints")
    plt.show()

    #############################################

if __name__ == "__main__":
    plot_linear_satellite()

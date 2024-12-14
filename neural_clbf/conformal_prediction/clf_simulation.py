import torch
from neural_clbf.controllers import NeuralCLBFController
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import pickle
import matplotlib
import matplotlib.pyplot as plt
from weighted_cp_quantile import *
import tqdm

matplotlib.use('TkAgg')

def clf_simulation(neural_controller, start_x, T):

    # Compute the number of simulations to run
    n_sims = start_x.shape[0]
    
    # Generate a tensor of start states
    n_dims = neural_controller.dynamics_model.n_dims
    n_controls = neural_controller.dynamics_model.n_controls
    x_sim_start = start_x

    # Make sure everything's on the right device
    device = "cpu"
    if hasattr(neural_controller, "device"):
        device = neural_controller.device  # type: ignore
    x_current = x_sim_start.to(device)

    # Simulate
    delta_t = neural_controller.dynamics_model.dt
    num_timesteps = int(T // delta_t)

    u_history = np.zeros((n_sims, n_controls, num_timesteps))
    r_history = np.zeros((n_sims, num_timesteps))
    x_history = np.zeros((n_sims, n_dims, num_timesteps))
    V_history = np.zeros((n_sims, num_timesteps))
    p_history = np.zeros((n_sims, num_timesteps))

    prog_bar_range = tqdm.trange(
        0, num_timesteps, desc = "CLF simulation", leave = True
    )
    
    for t in prog_bar_range: #range(num_timesteps):

        x_history[:,:,t] = x_current.cpu().detach().numpy()

        # Compute control input by solving the CLF-QP problem using the nominal (learned) model
        u_current, r_current = neural_controller.solve_CLF_QP(x_current)
        
        for i in range(n_sims):
            xdot = neural_controller.dynamics_model.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_current[i, :].unsqueeze(0)
            )
            u_history[i,:,t] = u_current[i, :].cpu().detach().numpy()
            r_history[i,t] = r_current[i].cpu().detach().item()

            V_current = neural_controller.V(x_current[i, :].unsqueeze(0))
            V_history[i,t] = V_current.cpu().detach().item()
            Lf_V, Lg_V = neural_controller.V_lie_derivatives(x_current[i, :].unsqueeze(0))
            clf_constraint = Lf_V + Lg_V @ u_current[i, :].T + neural_controller.clf_lambda * V_current
            p_history[i,t] = clf_constraint.cpu().detach().item()

            # Propagate the state
            x_current[i, :] = x_current[i, :] + delta_t * xdot.squeeze()

    # Plot
    fig, ax = plt.subplots(n_dims, 1)
    for d in range(n_dims):
        ax[d].plot(np.arange(num_timesteps) * delta_t, (x_history[:,d,:]).squeeze().T, color='blue')
        ax[d].grid(True)
        ax[d].set_ylabel("x [" + str(d) + "]")
    ax[n_dims-1].set_xlabel("Time (s)")
    ax[0].set_title("States")

    fig, ax = plt.subplots(2, 1)
    for i in range(n_sims):
        ax[0].plot(np.arange(num_timesteps) * delta_t, np.linalg.norm(x_history[i,:,:].squeeze().T, axis=1), color='blue')
    ax[0].set_ylabel("x 2-norm")
    ax[0].grid(True)
    for i in range(n_sims):
        ax[1].plot(np.arange(num_timesteps) * delta_t, V_history[i,:], color='blue')
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("V(x)")
    ax[1].grid(True)
    ax[0].set_title("State Norms and CLFs")

    fig, ax = plt.subplots(n_controls + 1, 1)
    for u in range(n_controls):
        for i in range(n_sims):
            ax[u].plot(np.arange(num_timesteps) * delta_t, u_history[i,u,:].squeeze().T, color='blue')
        ax[u].set_ylabel("u_QP [" + str(u) + "]")
        ax[u].grid(True)
    for i in range(n_sims):
        ax[n_controls].plot(np.arange(num_timesteps) * delta_t, r_history[i,:], color='blue')
    ax[n_controls].set_xlabel("Time (s)")
    ax[n_controls].set_ylabel("r_QP")
    ax[n_controls].grid(True)
    ax[0].set_title("QP Solver")

    fig, ax = plt.subplots(1, 1)
    for i in range(n_sims):
        ax.plot(np.arange(num_timesteps) * delta_t, p_history[i,:], color='blue')
    ax.set_ylabel("p = Vdot + c3*V")
    ax.set_xlabel("Time (s)")
    ax.grid(True)
    ax.set_title("CLF Constraints")

    plt.show()

if __name__ == "__main__":
    # Load the learned CLF
    #log_file = "./logs/inverted_pendulum_cart/commit_ca448fb/version_5/checkpoints/epoch=120-step=19880.ckpt" # nominal model
    log_file = "./logs/inverted_pendulum_cart_sindy/commit_604502a/version_0/checkpoints/epoch=51-step=43898.ckpt" # SINDy model
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Set up initial conditions for the sim
    # Update parameters
    start_x = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
        ]
    ) * 0.1

    # Run the sim
    clf_simulation(neural_controller, start_x, T = 1.0)
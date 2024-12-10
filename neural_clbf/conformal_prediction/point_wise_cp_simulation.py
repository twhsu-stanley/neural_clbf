import torch
from neural_clbf.controllers import NeuralCLBFController
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import pickle
import matplotlib
import matplotlib.pyplot as plt
from weighted_cp_quantile import *

matplotlib.use('TkAgg')

# TODO: Write a function that creates a CvxpyLayer solver
#       Write a function that simulates trajectories using clf_qp_cp

def create_clf_qp_cp_cvxpylayers_solver(neural_controller_cp):
    """Create a solver for the CLF-QP-CP problem using CvxpyLayer
    """

    # Define variables
    u = cp.Variable(neural_controller_cp.dynamics_model.n_controls)
    clf_relaxations = cp.Variable(1, nonneg=True)

    # Define the parameters that will be supplied at solve-time: the value
    # of the Lyapunov function, its Lie derivatives, the relaxation penalty, and
    # the reference control input
    Lf_V_params = cp.Parameter(1)
    Lg_V_params = cp.Parameter(neural_controller_cp.dynamics_model.n_controls)
    V_param = cp.Parameter(1, nonneg=True)
    cp_quantile_param = cp.Parameter(1, nonneg=True)
    u_ref_param = cp.Parameter(neural_controller_cp.dynamics_model.n_controls)
    clf_relaxation_penalty_param = cp.Parameter(1, nonneg=True)

    constraints = []
    # CLF decrease constraint (with relaxation)
    constraints.append(
        Lf_V_params
        + Lg_V_params @ u
        + neural_controller_cp.clf_lambda * V_param
        + cp_quantile_param
        - clf_relaxations
        <= 0
    )

    # Control limit constraints
    upper_lim, lower_lim = neural_controller_cp.dynamics_model.control_limits
    for control_idx in range(neural_controller_cp.dynamics_model.n_controls):
        constraints.append(u[control_idx] >= lower_lim[control_idx])
        constraints.append(u[control_idx] <= upper_lim[control_idx])

    # Define the objective
    objective_expression = cp.sum_squares(u - u_ref_param)
    objective_expression += cp.multiply(clf_relaxation_penalty_param, clf_relaxations)
    objective = cp.Minimize(objective_expression)

    # Create the optimization problem
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()
    variables = [u] + [clf_relaxations]
    parameters = [Lf_V_params] + [Lg_V_params]
    parameters += [V_param, cp_quantile_param, u_ref_param, clf_relaxation_penalty_param]
    clf_qp_cp_solver = CvxpyLayer(
        problem, variables=variables, parameters=parameters
    )

    return clf_qp_cp_solver

def clf_qp_cp_simulation(neural_controller, clf_qp_cp_solver, point_wise_cp_quantile, start_x, T):

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
    x_current_cp = x_sim_start.to(device)
    x_current_wcp = x_current_cp.clone() # using clone becuase tensors are pass-by-reference
    x_current_0 = x_current_cp.clone() 

    # Simulate
    delta_t = neural_controller.dynamics_model.dt / 5
    num_timesteps = int(T // delta_t)
    u_current_cp = torch.zeros(x_sim_start.shape[0], n_controls, device=device)
    u_current_wcp = torch.zeros(x_sim_start.shape[0], n_controls, device=device)
    u_current_0 = torch.zeros(x_sim_start.shape[0], n_controls, device=device)
    
    x_history_cp = np.zeros((n_sims, n_dims, num_timesteps))
    x_history_wcp = np.zeros((n_sims, n_dims, num_timesteps))
    x_history_0 = np.zeros((n_sims, n_dims, num_timesteps))
    V_history_cp = np.zeros((n_sims, num_timesteps))
    V_history_wcp = np.zeros((n_sims, num_timesteps))
    V_history_0 = np.zeros((n_sims, num_timesteps))
    p_history_cp = np.zeros((n_sims, num_timesteps)) # CLF constraint: p = Vdot + c3*V
    p_history_wcp = np.zeros((n_sims, num_timesteps))
    p_history_0 = np.zeros((n_sims, num_timesteps))
    Vdot_err_history_cp = np.zeros((n_sims, num_timesteps)) # Error term introduced by learning error
    Vdot_err_history_wcp = np.zeros((n_sims, num_timesteps))
    model_err_history_cp = np.zeros((n_sims, n_dims, num_timesteps))
    model_err_history_wcp = np.zeros((n_sims, n_dims, num_timesteps))
    cnstr_tightening_history_cp = np.zeros((n_sims, num_timesteps))
    cnstr_tightening_history_wcp = np.zeros((n_sims, num_timesteps))

    # Window for weighted CP
    window_size = 100 # number of datapoints
    wcp_alpha = 0.05
    rho = 0.995
    wcp_weights = rho**(np.arange(window_size,0,-1))
    wcp_weights = np.r_[wcp_weights,1] # weight n+1 should always be 1

    # Create a copy of neural_controller for each case
    neural_controller_cp = neural_controller
    neural_controller_wcp = neural_controller
    neural_controller_0 = neural_controller

    for t in range(num_timesteps):

        """1. Simulation using the ground truth model and non-weighted-CP-CLF-QP"""
        x_history_cp[:,:,t] = x_current_cp.cpu().detach().numpy()
        
        for i in range(n_sims):

            _, gradV_current = neural_controller_cp.V_with_jacobian(x_current_cp[i, :].unsqueeze(0))
            gradV_current = gradV_current.squeeze(0).cpu().detach().numpy()
            cnstr_tightening = np.linalg.norm(gradV_current.squeeze(), np.inf) * point_wise_cp_quantile # inf-norm * 1-norm

            # Compute control input by solving the CLF-QP-CP problem
            u_current_cp, _ = neural_controller_cp.u_CLF_QP_CP(x_current_cp[i, :].unsqueeze(0), clf_qp_cp_solver, cnstr_tightening)

            xdot = neural_controller_cp.dynamics_model.closed_loop_ground_truth_dynamics(
                x_current_cp[i, :].unsqueeze(0),
                u_current_cp
            )

            # Compute the errors
            f_ground_truth, g_ground_truth = neural_controller_cp.dynamics_model.control_affine_ground_truth_dynamics(x_current_cp[i, :].unsqueeze(0))
            f, g = neural_controller_cp.dynamics_model.control_affine_dynamics(x_current_cp[i, :].unsqueeze(0))
            f_ground_truth = f_ground_truth.squeeze(0).cpu().detach().numpy()
            g_ground_truth = g_ground_truth.squeeze(0).cpu().detach().numpy()
            f = f.squeeze(0).cpu().detach().numpy()
            g = g.squeeze(0).cpu().detach().numpy()
            uc = u_current_cp.unsqueeze(0).cpu().detach().numpy()

            model_err = f_ground_truth + g_ground_truth @ uc.T - f - g @ uc.T
            model_err_history_cp[i,:,t] = abs(model_err.squeeze())

            Vdot_err_history_cp[i,t] = (gradV_current @ model_err).item()

            Vdot_current = gradV_current @ (f_ground_truth + g_ground_truth @ uc.T)
            V_current = neural_controller_cp.V(x_current_cp[i, :].unsqueeze(0)).cpu().detach().item()
            V_history_cp[i,t] = V_current
            clf_constraint = Vdot_current.item() + neural_controller_cp.clf_lambda * V_current
            p_history_cp[i,t] = clf_constraint
            cnstr_tightening_history_cp[i,t] = cnstr_tightening

            # Propagate the state
            x_current_cp[i, :] = x_current_cp[i, :] + delta_t * xdot.squeeze()
        
        """2. Simulation using the ground truth model and weighted-CP-CLF-QP"""
        x_history_wcp[:,:,t] = x_current_wcp.cpu().detach().numpy()

        for i in range(n_sims):

            _, gradV_current = neural_controller_wcp.V_with_jacobian(x_current_wcp[i, :].unsqueeze(0))
            gradV_current = gradV_current.squeeze(0).cpu().detach().numpy()
            
            if t >= window_size:
                # sliding window of the 1-norm of modeling error as the nonconformity scores
                wcp_scores = np.sum(model_err_history_wcp[i,:,t-window_size:t], axis = 0) # 1-norm
                point_wise_wcp_quantile = weighted_cp_quantile(wcp_scores, wcp_weights[:window_size], wcp_alpha)
            else:
                point_wise_wcp_quantile = 0

            cnstr_tightening = np.linalg.norm(gradV_current.squeeze(), np.inf) * point_wise_wcp_quantile # inf-norm * 1-norm

            # Compute control input by solving the CLF-QP-CP problem
            u_current_wcp, _ = neural_controller_wcp.u_CLF_QP_CP(x_current_wcp[i, :].unsqueeze(0), clf_qp_cp_solver, cnstr_tightening)

            xdot = neural_controller_wcp.dynamics_model.closed_loop_ground_truth_dynamics(
                x_current_wcp[i, :].unsqueeze(0),
                u_current_wcp
            )

            # Compute the errors
            f_ground_truth, g_ground_truth = neural_controller_wcp.dynamics_model.control_affine_ground_truth_dynamics(x_current_wcp[i, :].unsqueeze(0))
            f, g = neural_controller_wcp.dynamics_model.control_affine_dynamics(x_current_wcp[i, :].unsqueeze(0))
            f_ground_truth = f_ground_truth.squeeze(0).cpu().detach().numpy()
            g_ground_truth = g_ground_truth.squeeze(0).cpu().detach().numpy()
            f = f.squeeze(0).cpu().detach().numpy()
            g = g.squeeze(0).cpu().detach().numpy()
            uc = u_current_wcp.unsqueeze(0).cpu().detach().numpy()

            model_err = f_ground_truth + g_ground_truth @ uc.T - f - g @ uc.T
            model_err_history_wcp[i,:,t] = abs(model_err.squeeze())

            Vdot_err_history_wcp[i,t] = (gradV_current @ model_err).item()

            Vdot_current = gradV_current @ (f_ground_truth + g_ground_truth @ uc.T)
            V_current = neural_controller_wcp.V(x_current_wcp[i, :].unsqueeze(0)).cpu().detach().item()
            V_history_wcp[i,t] = V_current
            clf_constraint = Vdot_current.item() + neural_controller_wcp.clf_lambda * V_current
            p_history_wcp[i,t] = clf_constraint
            cnstr_tightening_history_wcp[i,t] = cnstr_tightening

            # Propagate the state
            x_current_wcp[i, :] = x_current_wcp[i, :] + delta_t * xdot.squeeze()

        """3. Simulation using the ground truth model and non-CP CLF-QP """
        x_history_0[:,:,t] = x_current_0.cpu().detach().numpy()

        # Compute control input by solving the CLF-QP problem using the nominal (learned) model
        u_current_0 = neural_controller_0.u(x_current_0)

        for i in range(n_sims):
            xdot2 = neural_controller_0.dynamics_model.closed_loop_ground_truth_dynamics(
                x_current_0[i, :].unsqueeze(0),
                u_current_0[i, :].unsqueeze(0)
            )

            V_current2 = neural_controller_0.V(x_current_0[i, :].unsqueeze(0))
            V_history_0[i,t] = V_current2.cpu().detach().item()
            Lf_V, Lg_V = neural_controller_0.V_lie_derivatives(x_current_0[i, :].unsqueeze(0))
            clf_constraint = Lf_V + Lg_V @ u_current_0[i, :].T + neural_controller_0.clf_lambda * V_current2
            p_history_0[i,t] = clf_constraint.cpu().detach().item()

            # Propagate the state
            x_current_0[i, :] = x_current_0[i, :] + delta_t * xdot2.squeeze()

    # Plot
    fig, ax = plt.subplots(2, 1)
    for i in range(n_sims):
        ax[0].plot(np.arange(num_timesteps) * delta_t, np.linalg.norm(x_history_cp[i,:,:].squeeze().T, axis=1), color='red')
        ax[0].plot(np.arange(num_timesteps) * delta_t, np.linalg.norm(x_history_wcp[i,:,:].squeeze().T, axis=1), color='green')
        ax[0].plot(np.arange(num_timesteps) * delta_t, np.linalg.norm(x_history_0[i,:,:].squeeze().T, axis=1), color='blue')
    ax[0].set_ylabel("x 2-norm")
    ax[0].grid(True)
    for i in range(n_sims):
        ax[1].plot(np.arange(num_timesteps) * delta_t, V_history_cp[i,:], color='red')
        ax[1].plot(np.arange(num_timesteps) * delta_t, V_history_wcp[i,:], color='green')
        ax[1].plot(np.arange(num_timesteps) * delta_t, V_history_0[i,:], color='blue')
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("V(x)")
    ax[1].grid(True)

    fig, ax = plt.subplots(4, 1)
    for i in range(n_sims):
        ax[0].plot(np.arange(num_timesteps) * delta_t, p_history_cp[i,:], color='red')
        ax[0].plot(np.arange(num_timesteps) * delta_t, p_history_wcp[i,:], color='green')
        ax[0].plot(np.arange(num_timesteps) * delta_t, p_history_0[i,:], color='blue')
    #ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("p = Vdot + c3*V")
    ax[0].grid(True)
    #ax[0].set_title("CLF Constraint <? 0")
    #
    for i in range(n_sims):
        ax[1].plot(np.arange(num_timesteps) * delta_t, Vdot_err_history_cp[i,:], color='red')
        ax[1].plot(np.arange(num_timesteps) * delta_t, Vdot_err_history_wcp[i,:], color='green')
    #ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("gradV@Model Err")
    ax[1].grid(True)
    #ax[1].set_title("CLF Error")
    #
    for i in range(n_sims):
        ax[2].plot(np.arange(num_timesteps) * delta_t, model_err_history_cp[i,:,:].T)
    #ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("|f+g@u-f'-g'@u|")
    ax[2].grid(True)
    #ax[2].set_title("Model Error")
    for i in range(n_sims):
        ax[3].plot(np.arange(num_timesteps) * delta_t, cnstr_tightening_history_cp[i,:].T, color='red')
        ax[3].plot(np.arange(num_timesteps) * delta_t, cnstr_tightening_history_wcp[i,:].T, color='green')
    ax[3].set_xlabel("Time (s)")
    ax[3].set_ylabel("||gradV||*Quantile")
    ax[3].grid(True)
    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    # Load the learned CLF
    log_file = "./logs/inverted_pendulum_sindy/commit_4be3cd5/version_0/checkpoints/epoch=50-step=7190.ckpt" # constrained; training data with noise
    #log_file = "./logs/inverted_pendulum_sindy/commit_7e70ad1/version_0/checkpoints/epoch=50-step=7190.ckpt" # training data with noise
    #log_file = "./logs/inverted_pendulum_sindy/commit_c046f61/version_2/checkpoints/epoch=24-step=3524.ckpt" # training data without noise
    neural_controller_cp = NeuralCLBFController.load_from_checkpoint(log_file)

    # Load the saved CP quantile
    with open('./neural_clbf/conformal_prediction/quantiles/inverted_pendulum_sindy/' + 'point_wise_cp_quantile', 'rb') as file:
        point_wise_cp_quantile = pickle.load(file)

    # Create a QP solver
    clf_qp_cp_solver = create_clf_qp_cp_cvxpylayers_solver(neural_controller_cp)
    
    u1, _ = neural_controller_cp.u_CLF_QP_CP(torch.tensor([[0.0, 1.5]]), clf_qp_cp_solver, 0)
    u2 = neural_controller_cp.u(torch.tensor([[0.0, 1.5]]))
    assert u1 == u2

    # Set up initial conditions for the sim
    start_x = torch.tensor(
        [
            [1.9, 1.9],
            #[-0.9, 0.5],
            #[0.3, 1.5],
        ]
    )

    # Run the sim
    #point_wise_cp_quantile = 0.5
    clf_qp_cp_simulation(neural_controller_cp, clf_qp_cp_solver, point_wise_cp_quantile, start_x, T = 2)
from neural_clbf.controllers import NeuralCLBFController
import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import matplotlib
import matplotlib.pyplot as plt
import tqdm

matplotlib.use('TkAgg')

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
    cnstr_tightening = cp.Parameter(1, nonneg=True)
    u_ref_param = cp.Parameter(neural_controller_cp.dynamics_model.n_controls)
    clf_relaxation_penalty_param = cp.Parameter(1, nonneg=True)

    constraints = []
    # CLF decrease constraint (with relaxation)
    constraints.append(
        Lf_V_params
        + Lg_V_params @ u
        + neural_controller_cp.clf_lambda * V_param
        + cnstr_tightening
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
    parameters += [V_param, cnstr_tightening, u_ref_param, clf_relaxation_penalty_param]
    clf_qp_cp_solver = CvxpyLayer(
        problem, variables=variables, parameters=parameters
    )

    return clf_qp_cp_solver

def weighted_cp_quantile(R, cal_weights, alpha):
    """ Weighted Split Conformal Prediction"""

    # normalize weights (we add +1 in the denominator for the test point at n+1)
    weights_normalized = cal_weights / (np.sum(cal_weights)+1) # weight n+1 should always be 1

    if(np.sum(weights_normalized) >= 1-alpha):
        # calibration scores: |y_i - x_i @ betahat|
        #R = np.abs(y_cal - predictor.predict(X_cal))
        ord_R = np.argsort(R)
        # from when are the cumulative quantiles at least 1-\alpha
        ind_thresh = np.min(np.where(np.cumsum(weights_normalized[ord_R]) >= 1-alpha))
        # get the corresponding residual
        quantile = np.sort(R)[ind_thresh]
    else:
        quantile = np.inf
    
    # Standard prediction intervals using the absolute residual score quantile
    #mean_prediction = predictor.predict(X_test)
    #prediction_bands = np.stack([
    #    mean_prediction - quantile,
    #    mean_prediction + quantile
    #], axis=1)

    return quantile

def clf_simulation(neural_controller, clf_qp_cp_solver, start_x, T, solver_args = {"max_iters": 1000}):

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
        #u_current, r_current = neural_controller.solve_CLF_QP(x_current)
        u_current, r_current = neural_controller.u_CLF_QP_CP(x_current, clf_qp_cp_solver, 0.0, solver_args = solver_args)
        
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
        ax[d].plot(np.arange(num_timesteps) * delta_t, (x_history[:,d,:]).squeeze().T)
        ax[d].grid(True)
        ax[d].set_ylabel("x [" + str(d) + "]")
    ax[n_dims-1].set_xlabel("Time (s)")
    ax[0].set_title("States")

    fig, ax = plt.subplots(2, 1)
    for i in range(n_sims):
        ax[0].plot(np.arange(num_timesteps) * delta_t, np.linalg.norm(x_history[i,:,:].squeeze().T, axis=1))
    ax[0].set_ylabel("x 2-norm")
    ax[0].grid(True)
    for i in range(n_sims):
        ax[1].plot(np.arange(num_timesteps) * delta_t, V_history[i,:])
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("V(x)")
    ax[1].grid(True)
    ax[0].set_title("State Norms and CLFs")

    fig, ax = plt.subplots(n_controls + 1, 1)
    for u in range(n_controls):
        for i in range(n_sims):
            ax[u].plot(np.arange(num_timesteps) * delta_t, u_history[i,u,:].squeeze().T)
        ax[u].set_ylabel("u_QP [" + str(u) + "]")
        ax[u].grid(True)
    for i in range(n_sims):
        ax[n_controls].plot(np.arange(num_timesteps) * delta_t, r_history[i,:])
    ax[n_controls].set_xlabel("Time (s)")
    ax[n_controls].set_ylabel("r_QP")
    ax[n_controls].grid(True)
    ax[0].set_title("QP Solver")

    fig, ax = plt.subplots(1, 1)
    for i in range(n_sims):
        ax.plot(np.arange(num_timesteps) * delta_t, p_history[i,:])
    ax.set_ylabel("p = Vdot + lambda*V")
    ax.set_xlabel("Time (s)")
    ax.grid(True)
    ax.set_title("CLF Constraints")

    plt.show()

def clf_cp_simulation(neural_controller, clf_qp_cp_solver, cp_quantile, start_x, T, solver_args = {"max_iters": 1000}):

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
    r_history_cp = np.zeros((n_sims, num_timesteps))
    r_history_wcp = np.zeros((n_sims, num_timesteps))
    r_history_0 = np.zeros((n_sims, num_timesteps))
    model_err_history_wcp = np.zeros((n_sims, n_dims, num_timesteps))

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

    prog_bar_range = tqdm.trange(
        0, num_timesteps, desc = "CLF CP simulation", leave = True
    )
    
    for t in prog_bar_range: #range(num_timesteps):

        """1. Simulation using the ground truth model and non-weighted-CP-CLF-QP"""
        x_history_cp[:,:,t] = x_current_cp.cpu().detach().numpy()
        
        for i in range(n_sims):

            _, gradV_current = neural_controller_cp.V_with_jacobian(x_current_cp[i, :].unsqueeze(0))
            gradV_current = gradV_current.squeeze(0).cpu().detach().numpy()
            cnstr_tightening = np.linalg.norm(gradV_current.squeeze(), np.inf) * cp_quantile # inf-norm * 1-norm

            # Compute control input by solving the CLF-QP-CP problem
            u_current_cp, r_current_cp = neural_controller_cp.u_CLF_QP_CP(x_current_cp[i, :].unsqueeze(0), clf_qp_cp_solver, cnstr_tightening, 
                                                               solver_args = solver_args)

            xdot = neural_controller_cp.dynamics_model.closed_loop_ground_truth_dynamics(
                x_current_cp[i, :].unsqueeze(0),
                u_current_cp
            )

            f_ground_truth, g_ground_truth = neural_controller_cp.dynamics_model.control_affine_ground_truth_dynamics(x_current_cp[i, :].unsqueeze(0))
            f_ground_truth = f_ground_truth.squeeze(0).cpu().detach().numpy()
            g_ground_truth = g_ground_truth.squeeze(0).cpu().detach().numpy()
            uc = u_current_cp.unsqueeze(0).cpu().detach().numpy()

            Vdot_current = gradV_current @ (f_ground_truth + g_ground_truth @ uc.T)
            V_current = neural_controller_cp.V(x_current_cp[i, :].unsqueeze(0)).cpu().detach().item()
            V_history_cp[i,t] = V_current
            p_history_cp[i,t] = Vdot_current.item() + neural_controller_cp.clf_lambda * V_current
            r_history_cp[i,t] = r_current_cp.cpu().detach().item()

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
                wcp_quantile = weighted_cp_quantile(wcp_scores, wcp_weights[:window_size], wcp_alpha)
            else:
                wcp_quantile = 0.0

            cnstr_tightening = np.linalg.norm(gradV_current.squeeze(), np.inf) * wcp_quantile # inf-norm * 1-norm

            # Compute control input by solving the CLF-QP-CP problem
            u_current_wcp, r_current_wcp = neural_controller_wcp.u_CLF_QP_CP(x_current_wcp[i, :].unsqueeze(0), clf_qp_cp_solver, cnstr_tightening, 
                                                                 solver_args = solver_args)

            xdot = neural_controller_wcp.dynamics_model.closed_loop_ground_truth_dynamics(
                x_current_wcp[i, :].unsqueeze(0),
                u_current_wcp
            )

            f_ground_truth, g_ground_truth = neural_controller_wcp.dynamics_model.control_affine_ground_truth_dynamics(x_current_wcp[i, :].unsqueeze(0))
            f, g = neural_controller_wcp.dynamics_model.control_affine_dynamics(x_current_wcp[i, :].unsqueeze(0))
            f_ground_truth = f_ground_truth.squeeze(0).cpu().detach().numpy()
            g_ground_truth = g_ground_truth.squeeze(0).cpu().detach().numpy()
            f = f.squeeze(0).cpu().detach().numpy()
            g = g.squeeze(0).cpu().detach().numpy()
            uc = u_current_wcp.unsqueeze(0).cpu().detach().numpy()

            model_err = f_ground_truth + g_ground_truth @ uc.T - f - g @ uc.T
            model_err_history_wcp[i,:,t] = abs(model_err.squeeze())

            Vdot_current = gradV_current @ (f_ground_truth + g_ground_truth @ uc.T)
            V_current = neural_controller_wcp.V(x_current_wcp[i, :].unsqueeze(0)).cpu().detach().item()
            V_history_wcp[i,t] = V_current
            p_history_wcp[i,t] = Vdot_current.item() + neural_controller_wcp.clf_lambda * V_current
            r_history_wcp[i,t] = r_current_wcp.cpu().detach().item()

            # Propagate the state
            x_current_wcp[i, :] = x_current_wcp[i, :] + delta_t * xdot.squeeze()

        """3. Simulation using the ground truth model and non-CP CLF-QP """
        x_history_0[:,:,t] = x_current_0.cpu().detach().numpy()

        for i in range(n_sims):

            _, gradV_current = neural_controller_0.V_with_jacobian(x_current_0[i, :].unsqueeze(0))
            gradV_current = gradV_current.squeeze(0).cpu().detach().numpy()

            # Compute control input by solving the CLF-QP problem using the nominal (learned) model
            u_current_0, r_current_0 = neural_controller_0.u_CLF_QP_CP(x_current_0[i, :].unsqueeze(0), clf_qp_cp_solver, 0.0, 
                                                               solver_args = solver_args)

            xdot = neural_controller_0.dynamics_model.closed_loop_ground_truth_dynamics(
                x_current_0[i, :].unsqueeze(0),
                u_current_0
            )

            f_ground_truth, g_ground_truth = neural_controller_0.dynamics_model.control_affine_ground_truth_dynamics(x_current_0[i, :].unsqueeze(0))
            f_ground_truth = f_ground_truth.squeeze(0).cpu().detach().numpy()
            g_ground_truth = g_ground_truth.squeeze(0).cpu().detach().numpy()
            uc = u_current_0.unsqueeze(0).cpu().detach().numpy()

            Vdot_current = gradV_current @ (f_ground_truth + g_ground_truth @ uc.T)
            V_current = neural_controller_0.V(x_current_0[i, :].unsqueeze(0)).cpu().detach().item()
            V_history_0[i,t] = V_current
            p_history_0[i,t] = Vdot_current.item() + neural_controller_0.clf_lambda * V_current
            r_history_0[i,t] = r_current_0.cpu().detach().item()

            # Propagate the state
            x_current_0[i, :] = x_current_0[i, :] + delta_t * xdot.squeeze()

    # Plot
    fig, ax = plt.subplots(n_dims, 1)
    for d in range(n_dims):
        ax[d].plot(np.arange(num_timesteps) * delta_t, (x_history_cp[:,d,:]).squeeze().T, color='red')
        ax[d].plot(np.arange(num_timesteps) * delta_t, (x_history_wcp[:,d,:]).squeeze().T, color='green')
        ax[d].plot(np.arange(num_timesteps) * delta_t, (x_history_0[:,d,:]).squeeze().T, color='blue')
        ax[d].grid(True)
        ax[d].set_ylabel("x [" + str(d) + "]")
    ax[n_dims-1].set_xlabel("Time (s)")
    ax[0].set_title("States")

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

    fig, ax = plt.subplots(1, 1)
    for i in range(n_sims):
        ax.plot(np.arange(num_timesteps) * delta_t, p_history_cp[i,:], color='red')
        ax.plot(np.arange(num_timesteps) * delta_t, p_history_wcp[i,:], color='green')
        ax.plot(np.arange(num_timesteps) * delta_t, p_history_0[i,:], color='blue')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("p = Vdot + lambda*V")
    ax.grid(True)

    fig, ax = plt.subplots(1, 1)
    for i in range(n_sims):
        ax.plot(np.arange(num_timesteps) * delta_t, r_history_cp[i,:], color='red')
        ax.plot(np.arange(num_timesteps) * delta_t, r_history_wcp[i,:], color='green')
        ax.plot(np.arange(num_timesteps) * delta_t, r_history_0[i,:], color='blue')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("QP relaxation")
    ax.grid(True)

    fig.tight_layout()

    plt.show()

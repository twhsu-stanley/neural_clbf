import torch
from neural_clbf.controllers import NeuralCLBFController
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import pickle
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# TODO: Write a function that creates a CvxpyLayer solver
#       Write a function that simulates trajectories using clf_qp_cp

def create_clf_qp_cp_cvxpylayers_solver(neural_controller):
    """Create a solver for the CLF-QP-CP problem using CvxpyLayer
    """

    # Define variables
    u = cp.Variable(neural_controller.dynamics_model.n_controls)
    clf_relaxations = cp.Variable(1, nonneg=True)

    # Define the parameters that will be supplied at solve-time: the value
    # of the Lyapunov function, its Lie derivatives, the relaxation penalty, and
    # the reference control input
    Lf_V_params = cp.Parameter(1)
    Lg_V_params = cp.Parameter(neural_controller.dynamics_model.n_controls)
    V_param = cp.Parameter(1, nonneg=True)
    cp_quantile_param = cp.Parameter(1, nonneg=True)
    u_ref_param = cp.Parameter(neural_controller.dynamics_model.n_controls)
    clf_relaxation_penalty_param = cp.Parameter(1, nonneg=True)

    constraints = []
    # CLF decrease constraint (with relaxation)
    constraints.append(
        Lf_V_params
        + Lg_V_params @ u
        + neural_controller.clf_lambda * V_param
        + cp_quantile_param
        - clf_relaxations
        <= 0
    )

    # Control limit constraints
    upper_lim, lower_lim = neural_controller.dynamics_model.control_limits
    for control_idx in range(neural_controller.dynamics_model.n_controls):
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
    x_current = x_sim_start.to(device)
    x_current2 = x_sim_start.to(device)

    # Simulate
    delta_t = neural_controller.dynamics_model.dt
    num_timesteps = int(T // delta_t)
    u_current = torch.zeros(x_sim_start.shape[0], n_controls, device=device)
    u_current2 = torch.zeros(x_sim_start.shape[0], n_controls, device=device)
    
    x_history = np.zeros((n_sims, n_dims, num_timesteps))
    x_history2 = np.zeros((n_sims, n_dims, num_timesteps))

    for t in range(num_timesteps):
        # Compute control input by solving the CLF-QP-CP problem
        u_current, _ = neural_controller.u_CLF_QP_CP(x_current, clf_qp_cp_solver, point_wise_cp_quantile)
        #u_current = neural_controller.u(x_current)

        # Simulate forward using the dynamics
        for i in range(n_sims):
            # Use the ground truth model
            xdot = neural_controller.dynamics_model.closed_loop_ground_truth_dynamics(
                x_current[i, :].unsqueeze(0),
                u_current[i, :].unsqueeze(0)
            )
            x_current[i, :] = x_current[i, :] + delta_t * xdot.squeeze()

        x_history[:,:,t] = x_current.cpu().detach().numpy()

        # Compute control input by solving the CLF-QP-CP problem
        u_current2, _ = neural_controller.u_CLF_QP_CP(x_current2, clf_qp_cp_solver, point_wise_cp_quantile)

        # Simulate forward using the dynamics
        for i in range(n_sims):
            # Use the learned model
            xdot2 = neural_controller.dynamics_model.closed_loop_dynamics(
                x_current2[i, :].unsqueeze(0),
                u_current2[i, :].unsqueeze(0)
            )
            x_current2[i, :] = x_current2[i, :] + delta_t * xdot2.squeeze()

        x_history2[:,:,t] = x_current2.cpu().detach().numpy()

    # Plot
    plt.figure("State 2-norm")
    for i in range(n_sims):
        plt.plot(np.arange(num_timesteps) * delta_t, np.linalg.norm(x_history[i,:,:].squeeze().T, axis=1), color='red')
        plt.plot(np.arange(num_timesteps) * delta_t, np.linalg.norm(x_history2[i,:,:].squeeze().T, axis=1), color='blue')
    plt.show()

if __name__ == "__main__":
    # Load the learned CLF
    log_file = "./logs/inverted_pendulum_sindy/commit_c046f61/version_2/checkpoints/epoch=24-step=3524.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Load the saved CP quantile
    with open('./neural_clbf/conformal_prediction/quantiles/inverted_pendulum_sindy/' + 'point_wise_cp_quantile', 'rb') as file:
        point_wise_cp_quantile = pickle.load(file)

    # Create a QP solver
    clf_qp_cp_solver = create_clf_qp_cp_cvxpylayers_solver(neural_controller)
    
    u1, _ = neural_controller.u_CLF_QP_CP(torch.tensor([[0.0, 1.5]]), clf_qp_cp_solver, 0)
    u2 = neural_controller.u(torch.tensor([[0.0, 1.5]]))
    assert u1 == u2

    # Set up initial conditions for the sim
    start_x = torch.tensor(
        [
            [1.5, 1.5],
            #[0.9, 1.5],
            [0.3, 1.5],
        ]
    )

    # Run the sim
    clf_qp_cp_simulation(neural_controller, clf_qp_cp_solver, point_wise_cp_quantile, start_x, T = 3)
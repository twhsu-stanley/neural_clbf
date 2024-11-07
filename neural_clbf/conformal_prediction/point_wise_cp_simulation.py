import torch
from neural_clbf.controllers import NeuralCLBFController
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


# Load the learned CLF
log_file = "./logs/inverted_pendulum_sindy/commit_c046f61/version_2/checkpoints/epoch=24-step=3524.ckpt"
neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

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
differentiable_qp_cp_solver = CvxpyLayer(
    problem, variables=variables, parameters=parameters
)


neural_controller.u_CLF_QP_CP(torch.zeros(2, 2), differentiable_qp_cp_solver, torch.zeros(1))

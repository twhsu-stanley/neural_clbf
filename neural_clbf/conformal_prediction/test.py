from point_wise_cp_simulation import *

# Load the learned CLF
log_file = "./logs/inverted_pendulum_cart_sindy/commit_ca448fb/version_24/checkpoints/epoch=120-step=19880.ckpt" # SINDy model
neural_controller_cp = NeuralCLBFController.load_from_checkpoint(log_file)
neural_controller_cp.dynamics_model.dt = 0.001

# Create a QP solver
clf_qp_cp_solver = create_clf_qp_cp_cvxpylayers_solver(neural_controller_cp)

u1, _ = neural_controller_cp.u_CLF_QP_CP(torch.tensor([[0.0, 1.5, 1.2, 3.1]]), clf_qp_cp_solver, 0)
u2 = neural_controller_cp.u(torch.tensor([[0.0, 1.5, 1.2, 3.1]]))
assert u1 == u2

# Set up initial conditions for the sim
start_x = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0],
    ]
)

# Run the sim
point_wise_cp_quantile = 0.0
clf_qp_cp_simulation(neural_controller_cp, clf_qp_cp_solver, point_wise_cp_quantile, start_x, T = 2)
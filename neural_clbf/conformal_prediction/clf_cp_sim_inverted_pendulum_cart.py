import torch
from neural_clbf.controllers import NeuralCLBFController
import pickle
from neural_clbf.conformal_prediction.clf_cp_sim_utils import *

if __name__ == "__main__":
    # Load the learned CLF
    
    # Nominal [worked w/ run iter = 1000]: 4L; 1e4 samples; train iter = 1000
    #log_file = "./logs/inverted_pendulum_cart/4L_1e4s/version_1/checkpoints/epoch=120-step=19880.ckpt"
    
    # SINDy [worked w/ run iter = 1000]: 4L; 1e4 samples; train iter = 1000
    #log_file = "./logs/inverted_pendulum_cart_sindy/4L_1e4s/version_0/checkpoints/epoch=120-step=19880.ckpt"
    
    # SINDy [worked w/ run iter = 500]: 5L; 1e4 samples; train iter = 1000 
    #log_file = "./logs/inverted_pendulum_cart_sindy/5L_1e4s/version_0/checkpoints/epoch=120-step=19880.ckpt"
    
    # SINDy [worked w/ run iter = ?]: 4L; 2e4 samples; train iter = 1000
    #log_file = "./logs/inverted_pendulum_cart_sindy/4L_2e4s/version_1/checkpoints/epoch=150-step=56631.ckpt"

    # SINDy [worked w/ run iter = 500]: 4L; 1e4 samples; train iter = 1000; CLBF with roa_reg and positive_loss
    log_file = "./logs/inverted_pendulum_cart_sindy/4L_1e4s_roa_regulator/version_2/checkpoints/epoch=120-step=19880.ckpt"
    
    neural_controller_cp = NeuralCLBFController.load_from_checkpoint(log_file)

    #neural_controller_cp.clf_lambda = 0.0
    #neural_controller_cp.dynamics_model.nominal_params = {'M': 1.0, 'm': 1.0, 'L': 0.5, 'Kd': 10.0}
    #{"M": 1.0, "m": 1.0, "L": 0.5, "Kd": 10.0}

    # Load CP quantile
    # TODO: make model_error a member of the model
    with open('../pysindy/control_affine_models/saved_models/model_inverted_pendulum_cart_sindy', 'rb') as file:
        model = pickle.load(file)
    cp_quantile = model.model_error['quantile']
    cp_alpha = model.model_error['alpha']
    print("CP alpha = %4.2f; CP quantile = %5.3f" % (cp_alpha, cp_quantile))

    # Create a QP solver
    clf_qp_cp_solver = create_clf_qp_cp_cvxpylayers_solver(neural_controller_cp)
    
    u1, _ = neural_controller_cp.u_CLF_QP_CP(torch.tensor([[2.0, -1.0, 1.5, -2.0]]), clf_qp_cp_solver, 0.0, solver_args = {"max_iters": 1000})
    u2 = neural_controller_cp.u(torch.tensor([[2.0, -1.0, 1.5, -2.0]]))
    assert u1 == u2

    # Set up initial conditions for the sim
    # TODO: this is model-specific; make it general
    start_x = torch.tensor(
        [
            [-1.0, -5.0, 0.0, -5.0],
            #[0.0, 0.0, -0.8, 0.0],
        ]
    )

    # Run the sim
    #cp_quantile = 0.5
    clf_cp_simulation(neural_controller_cp, clf_qp_cp_solver, cp_quantile, start_x, T = 3.0, solver_args = {"max_iters": 500})
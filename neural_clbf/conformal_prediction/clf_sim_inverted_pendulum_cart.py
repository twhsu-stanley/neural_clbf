import torch
from neural_clbf.controllers import NeuralCLBFController
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
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

    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    neural_controller.clf_lambda = 0.0

    # Create a QP solver
    clf_qp_solver = create_clf_qp_cp_cvxpylayers_solver(neural_controller)

    # Test controller
    x_test = torch.tensor([[2.0, -1.0, 1.5, -2.0]])
    u_1 = neural_controller.solve_CLF_QP(x_test)
    u_2 = neural_controller.u_CLF_QP_CP(x_test, clf_qp_solver, 0.0, solver_args = {"max_iters": 1000})
    #assert u_1 == u_2

    # Set up initial conditions for the sim
    # TODO: this is model-specific; make it general
    start_x = torch.tensor(
        [
            #[-1.0, -5.0, 0.0, -5.0],
            [0.0, 0.0, 0.8, 0.0],
            [0.0, 0.0, -0.8, 0.0],
            [0.0, 0.0, 0.9, 0.0],
            [0.0, 0.0, -0.9, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )

    # Run the sim
    #solver_args = {"solve_method": "ECOS", "max_iters": 10000, "reltol": 1e-8}
    #solver_args = {"eps": 1e-8, "max_iters": 10000, "acceleration_lookback": 0}
    solver_args = {"max_iters": 500}
    clf_simulation(neural_controller,  clf_qp_solver, start_x, T = 3.0, solver_args = solver_args)
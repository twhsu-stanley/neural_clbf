import torch
from neural_clbf.controllers import NeuralCLBFController
import pickle
import dill
import numpy as np

# TODO: Generic functions that calculates and returns the CP quantiles for the point-wise case and the finite-horizon case

def calc_point_wise_cp_quantile(neural_controller, sindy_model, trajectory_data, cp_alpha):
    """Perform point-wise conformal prediction and return the quantile
    """

    # Get calibration set and validation set 
    x_cal = trajectory_data["x_cal"]
    u_cal = trajectory_data["u_cal"]
    x_val = trajectory_data["x_val"]
    u_val = trajectory_data["u_val"]
    dt = trajectory_data["dt"]
    
    # Conpute nonconformity score for <JV(x), f(x,u)-\hat{f}(x,u)>
    nc_score = []
    for i in range(len(x_cal)):
        model_err = (sindy_model.predict(x_cal[i], u = u_cal[i]) - sindy_model.differentiate(x_cal[i], t = dt))
        _, JV = neural_controller.V_with_jacobian(torch.tensor(x_cal[i], dtype = torch.float32))
        JV = JV.detach().numpy()
        for j in range(x_cal[i].shape[0]):
            R = JV[j,:,:] @ model_err[j,:]
            nc_score.extend(R)

    n = len(nc_score)
    cp_quantile = np.quantile(nc_score, np.ceil((n + 1) * (1 - cp_alpha)) / n, interpolation="higher")

    return cp_quantile

if __name__ == "__main__":
    # Load the learned CLF
    log_file = "./logs/inverted_pendulum_sindy/commit_4be3cd5/version_0/checkpoints/epoch=50-step=7190.ckpt" # constrained; training data with noise
    #log_file = "./logs/inverted_pendulum_sindy/commit_7e70ad1/version_0/checkpoints/epoch=50-step=7190.ckpt" # training data with noise
    #log_file = "./logs/inverted_pendulum_sindy/commit_c046f61/version_2/checkpoints/epoch=24-step=3524.ckpt" # training data without noise
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Load the SINDY model
    with open('../pysindy/control_affine_models/saved_models/model_inverted_pendulum_sindy', 'rb') as file:
        sindy_model = dill.load(file)

    # Load trajectory data (calibration and validation sets)
    with open('../pysindy/control_affine_models/trajectory_data/traj_inverted_pendulum_sindy', 'rb') as file:
        trajectory_data = pickle.load(file)

    cp_alpha = 0.05
    point_wise_cp_quantile = calc_point_wise_cp_quantile(neural_controller, sindy_model, trajectory_data, cp_alpha)

    with open('./neural_clbf/conformal_prediction/quantiles/inverted_pendulum_sindy/' + 'point_wise_cp_quantile', 'wb') as file:
	    pickle.dump(point_wise_cp_quantile, file)

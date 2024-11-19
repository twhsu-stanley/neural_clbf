import torch
from neural_clbf.controllers import NeuralCLBFController
import pickle
import dill
import numpy as np

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
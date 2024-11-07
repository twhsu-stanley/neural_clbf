import torch
from neural_clbf.controllers import NeuralCLBFController
import pickle
import numpy as np

def get_cp_inverted_pendulum_sindy():
    # Load the learned CLF
    log_file = "./logs/inverted_pendulum_sindy/commit_c046f61/version_2/checkpoints/epoch=24-step=3524.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Load the SINDY model
    with open('../pysindy/control_affine_models/saved_models/model_inverted_pendulum_sindy', 'rb') as file:
        model = pickle.load(file)

    # Load trajectory data (calibration and validation sets)
    with open('../pysindy/control_affine_models/trajectory_data/traj_inverted_pendulum_sindy', 'rb') as file:
        trajectory_data = pickle.load(file)
        x_cal = trajectory_data["x_cal"]
        u_cal = trajectory_data["u_cal"]
        x_val = trajectory_data["x_val"]
        u_val = trajectory_data["u_val"]
        dt = trajectory_data["dt"]
    
    # Conpute nonconformity score for <JV(x), f(x,u)-\hat{f}(x,u)>
    # Point-wise Conformal Prediction
    nc_score = []
    for i in range(len(x_cal)):
        model_err = (model.predict(x_cal[i], u = u_cal[i]) - model.differentiate(x_cal[i], t = dt))
        _, JV = neural_controller.V_with_jacobian(torch.tensor(x_cal[i], dtype = torch.float32))
        JV = JV.detach().numpy()
        for j in range(x_cal[i].shape[0]):
            R = JV[j,:,:] @ model_err[j,:]
            nc_score.extend(R)

    alpha = 0.05
    n = len(nc_score)
    cp_quantile = np.quantile(nc_score, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher")

    # TODO: The above stpes should be incorporated into the training script immediately after trainer.fit(clbf_controller)
    # TODO: Use neural_controller.set_cp_quantile(cp_quantile) before running experiment 

if __name__ == "__main__":
    get_cp_inverted_pendulum_sindy()

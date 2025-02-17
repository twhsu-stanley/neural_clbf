"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List
from math import sqrt

import torch
import numpy as np
import pickle

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList, predict_tensor

# Load the SINDy model ##########################################################################
with open('../control_affine_models/saved_models/model_dubins_car_sindy', 'rb') as file:
    model = pickle.load(file)

feature_names = model.get_feature_names()
n_features = len(feature_names)
for i in range(n_features):
    feature_names[i] = feature_names[i].replace(" ", "*")
    feature_names[i] = feature_names[i].replace("^", "**")
    feature_names[i] = feature_names[i].replace("sin", "torch.sin")
    feature_names[i] = feature_names[i].replace("cos", "torch.cos")

coefficients = model.optimizer.coef_

# Get indices of the SINDy regressor corresponding to each state and control input
idx_x = [] # Indices for f(x)
idx_u = [] # Indices for g(x)*u
for i in range(len(feature_names)):
    if 'u0' in feature_names[i]:
        idx_u.append(i)
    else:
        idx_x.append(i)

cp_quantile = model.model_error['quantile']
#################################################################################################

class LinearSatelliteSINDy(ControlAffineSystem):
    """
    Represents a satellite through the linearized Clohessy-Wiltshire equations

    The system has state

        x = [x, y, z, xdot, ydot, zdot]

    representing the position and velocity of the chaser satellite, and it
    has control inputs

        u = [ux, uy, uz]

    representing the thrust applied in each axis. Distances are in km, and control
    inputs are measured in km/s^2.

    The task here is to get to the origin without leaving the bounding box [-5, 5] on
    all positions and [-1, 1] on velocities.

    The system is parameterized by
        a: the length of the semi-major axis of the target's orbit (e.g. 6871)
        ux_target, uy_target, uz_target: accelerations due to unmodelled effects and
                                         target control.
    """

    # Number of states and controls
    N_DIMS = 6
    N_CONTROLS = 3

    # State indices
    X = 0
    Y = 1
    Z = 2
    XDOT = 3
    YDOT = 4
    ZDOT = 5
    # Control indices
    UX = 0
    UY = 1
    UZ = 2

    # Constant parameters
    MU = 3.986e14  # Earth's gravitational parameter

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
        use_l1_norm: bool = False,
    ):
        """
        Initialize the inverted pendulum.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
            use_l1_norm: if True, use L1 norm for safety zones; otherwise, use L2
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios
        )
        self.use_l1_norm = use_l1_norm

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # Make sure all needed parameters were provided
        valid = valid and "a" in params
        valid = valid and "ux_target" in params
        valid = valid and "uy_target" in params
        valid = valid and "uz_target" in params

        # Make sure all parameters are physically valid
        valid = valid and params["a"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return LinearSatelliteSINDy.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return LinearSatelliteSINDy.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[LinearSatelliteSINDy.X] = 2.0
        upper_limit[LinearSatelliteSINDy.Y] = 2.0
        upper_limit[LinearSatelliteSINDy.Z] = 2.0
        upper_limit[LinearSatelliteSINDy.XDOT] = 1
        upper_limit[LinearSatelliteSINDy.YDOT] = 1
        upper_limit[LinearSatelliteSINDy.ZDOT] = 1

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([1.0, 1.0, 1.0])
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Stay within some maximum distance from the target
        order = 1 if hasattr(self, "use_l1_norm") and self.use_l1_norm else 2
        distance = x[:, : LinearSatelliteSINDy.Z + 1].norm(dim=-1, p=order)
        # safe_mask.logical_and_(distance <= 1.0)

        # Stay at least some minimum distance from the target
        safe_mask.logical_and_(distance >= 0.75)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # Maximum distance
        order = 1 if hasattr(self, "use_l1_norm") and self.use_l1_norm else 2
        distance = x[:, : LinearSatelliteSINDy.Z + 1].norm(dim=-1, p=order)
        # unsafe_mask.logical_or_(distance >= 1.5)

        # Minimum distance
        unsafe_mask.logical_or_(distance <= 0.25)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of points in the state space
        """
        order = 1 if hasattr(self, "use_l1_norm") and self.use_l1_norm else 2
        goal_mask = x[:, : LinearSatelliteSINDy.Z + 1].norm(dim=-1, p=order) <= 0.5

        return goal_mask

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # Compute f(x) using the SINDy model
        f_of_x = predict_tensor(x, torch.zeros((batch_size,1)), self.feature_names, self.coefficients, self.idx_x)

        f[:, LinearSatelliteSINDy.X, 0] = f_of_x[:,0]
        f[:, LinearSatelliteSINDy.Y, 0] = f_of_x[:,1]
        f[:, LinearSatelliteSINDy.Z, 0] = f_of_x[:,2]
        f[:, LinearSatelliteSINDy.XDOT, 0] = f_of_x[:,3]
        f[:, LinearSatelliteSINDy.YDOT, 0] = f_of_x[:,4]
        f[:, LinearSatelliteSINDy.ZDOT, 0] = f_of_x[:,5]

        return f

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        # Compute g(x) using the SINDy model
        for index_u in range(self.n_controls):
            u_zero = torch.zeros((batch_size, self.n_controls))
            u_zero[:,index_u] = 1
            g_of_x = predict_tensor(x, u_zero, self.feature_names, self.coefficients, feature_indices = self.idx_u)

            g[:, LinearSatelliteSINDy.X, index_u] = g_of_x[:,0]
            g[:, LinearSatelliteSINDy.Y, index_u] = g_of_x[:,1]
            g[:, LinearSatelliteSINDy.Z, index_u] = g_of_x[:,2]
            g[:, LinearSatelliteSINDy.XDOT, index_u] = g_of_x[:,3]
            g[:, LinearSatelliteSINDy.YDOT, index_u] = g_of_x[:,4]
            g[:, LinearSatelliteSINDy.ZDOT, index_u] = g_of_x[:,5]

        return g
    
    def _f_ground_truth(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # Extract the needed parameters
        a = params["a"]
        ux_target = params["ux_target"]
        uy_target = params["uy_target"]
        uz_target = params["uz_target"]
        # Compute mean-motion
        n = sqrt(LinearSatelliteSINDy.MU / a ** 3)
        # and state variables
        x_ = x[:, LinearSatelliteSINDy.X]
        z_ = x[:, LinearSatelliteSINDy.Z]
        xdot_ = x[:, LinearSatelliteSINDy.XDOT]
        ydot_ = x[:, LinearSatelliteSINDy.YDOT]
        zdot_ = x[:, LinearSatelliteSINDy.ZDOT]

        # The first three dimensions just integrate the velocity
        f[:, LinearSatelliteSINDy.X, 0] = xdot_
        f[:, LinearSatelliteSINDy.Y, 0] = ydot_
        f[:, LinearSatelliteSINDy.Z, 0] = zdot_

        # The last three use the CHW equations
        f[:, LinearSatelliteSINDy.XDOT, 0] = 3 * n ** 2 * x_ + 2 * n * ydot_
        f[:, LinearSatelliteSINDy.YDOT, 0] = -2 * n * xdot_
        f[:, LinearSatelliteSINDy.ZDOT, 0] = -(n ** 2) * z_

        # Add perturbations
        f[:, LinearSatelliteSINDy.XDOT, 0] += ux_target
        f[:, LinearSatelliteSINDy.YDOT, 0] += uy_target
        f[:, LinearSatelliteSINDy.ZDOT, 0] += uz_target

        return f

    def _g_ground_truth(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        # The control inputs are accelerations
        g[:, LinearSatelliteSINDy.XDOT, LinearSatelliteSINDy.UX] = 1.0
        g[:, LinearSatelliteSINDy.YDOT, LinearSatelliteSINDy.UY] = 1.0
        g[:, LinearSatelliteSINDy.ZDOT, LinearSatelliteSINDy.UZ] = 1.0

        return g
"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List
from math import sqrt

import torch
import numpy as np
import pickle, dill

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList, predict_tensor

# Load the SINDy model ##########################################################################
with open('./SINDy_models/model_dubins_car_traj_sindy', 'rb') as file:
    model = pickle.load(file)

feature_names = model["feature_names"]
n_features = len(feature_names)
for i in range(n_features):
    feature_names[i] = feature_names[i].replace(" ", "*")
    feature_names[i] = feature_names[i].replace("^", "**")
    feature_names[i] = feature_names[i].replace("sin", "torch.sin")
    feature_names[i] = feature_names[i].replace("cos", "torch.cos")

coefficients = model["coefficients"]

idx_x = [] # Indices for f(x)
idx_u = [] # Indices for g(x)*u
for i in range(len(feature_names)):
    if 'u0' in feature_names[i]:
        idx_u.append(i)
    else:
        idx_x.append(i)
        
cp_quantile = model["model_error"]['quantile']
#print("cp_quantile = ", cp_quantile)
#################################################################################################

class DubinsCarSINDy(ControlAffineSystem):
    """
    """

    # Number of states and controls
    N_DIMS = 3
    N_CONTROLS = 1

    # State indices
    X = 0
    Y = 1
    THETA = 2
    
    # Control indices
    U = 0

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
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
        self.feature_names = feature_names
        self.coefficients = coefficients
        self.idx_x = idx_x
        self.idx_u = idx_u
        self.cp_quantile = cp_quantile
        
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios
        )

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # Make sure all needed parameters were provided
        valid = valid and "v" in params

        # Make sure all parameters are physically valid
        valid = valid and params["v"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return DubinsCarSINDy.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return DubinsCarSINDy.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[DubinsCarSINDy.X] = 5.0
        upper_limit[DubinsCarSINDy.Y] = 5.0
        upper_limit[DubinsCarSINDy.THETA] = 100/180*np.pi

        lower_limit = torch.ones(self.n_dims)
        lower_limit[DubinsCarSINDy.X] = -5.0
        lower_limit[DubinsCarSINDy.Y] = -5.0
        lower_limit[DubinsCarSINDy.THETA] = -100/180*np.pi

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([100])
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)
    
    @property
    def goal_point(self):
        return torch.tensor([[4.5, 0.0, 0.0]])
    
    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        d = torch.pow(x[:,DubinsCarSINDy.X] - 0.0, 2) + torch.pow(x[:,DubinsCarSINDy.Y] - 0.0, 2) - 4
        #v = self.nominal_params['v']
        #drate =  (2 * (x[:,DubinsCarSINDy.X] - 5.0) * v * torch.cos(x[:,DubinsCarSINDy.THETA]) + 2 * (x[:,DubinsCarSINDy.Y] - 4.0) * v * torch.sin(x[:,DubinsCarSINDy.THETA]))
        #h = 15 * d + drate
        #safe_mask.logical_and_(h >= 33.75)

        # TODO: Try this new safe set
        safe_mask.logical_and_(d >= 1.0)
        safe_mask.logical_and_(x[:,DubinsCarSINDy.THETA] <= 70/180*np.pi)
        safe_mask.logical_and_(x[:,DubinsCarSINDy.THETA] >= -70/180*np.pi)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        d = torch.pow(x[:,DubinsCarSINDy.X] - 0.0, 2) + torch.pow(x[:,DubinsCarSINDy.Y] - 0.0, 2) - 4
        #v = self.nominal_params['v']
        #drate =  (2 * (x[:,DubinsCarSINDy.X] - 5.0) * v * torch.cos(x[:,DubinsCarSINDy.THETA]) + 2 * (x[:,DubinsCarSINDy.Y] - 4.0) * v * torch.sin(x[:,DubinsCarSINDy.THETA]))
        #h = 15 * d + drate
        #unsafe_mask.logical_or_(h < 0)

        # TODO: Try this new safe set
        unsafe_mask.logical_or_(d <= 0)
        unsafe_mask.logical_or_(x[:,DubinsCarSINDy.THETA] > np.pi/2)
        unsafe_mask.logical_or_(x[:,DubinsCarSINDy.THETA] < -np.pi/2)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of points in the state space
        """
        # TODO: verify if the goal_mask is not used at all
        goal_mask = (torch.pow(x[:,DubinsCarSINDy.X] - 4.5, 2) + torch.pow(x[:,DubinsCarSINDy.Y] - 0.0, 2) <= 0.1)

        return goal_mask

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
        v = params["v"]
        
        # and state variables
        #x_ = x[:, DubinsCarSINDy.X]
        #y_ = x[:, DubinsCarSINDy.Y]
        theta_ = x[:, DubinsCarSINDy.THETA]

        f[:, DubinsCarSINDy.X, 0] = v * torch.cos(theta_)
        f[:, DubinsCarSINDy.Y, 0] = v * torch.sin(theta_)
        f[:, DubinsCarSINDy.THETA, 0] = 0.0

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
        g[:, DubinsCarSINDy.X, DubinsCarSINDy.U] = 0.0
        g[:, DubinsCarSINDy.Y, DubinsCarSINDy.U] = 0.0
        g[:, DubinsCarSINDy.THETA, DubinsCarSINDy.U] = 1.0

        return g
    
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

        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # Compute f(x) using the SINDy model
        # TODO
        f_of_x = predict_tensor(x, torch.zeros((batch_size,1)), self.feature_names, self.coefficients, self.idx_x)
        
        f[:, DubinsCarSINDy.X, 0] = f_of_x[:,0]
        f[:, DubinsCarSINDy.Y, 0] = f_of_x[:,1]
        f[:, DubinsCarSINDy.THETA, 0] = f_of_x[:,2]

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

        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        # Compute g(x) using the SINDy model
        # TODO
        g_of_x = predict_tensor(x, torch.ones((batch_size,1)), self.feature_names, self.coefficients, feature_indices = self.idx_u)

        # Effect on theta dot
        g[:, DubinsCarSINDy.X, DubinsCarSINDy.U] = g_of_x[:,0]
        g[:, DubinsCarSINDy.Y, DubinsCarSINDy.U] = g_of_x[:,1]
        g[:, DubinsCarSINDy.THETA, DubinsCarSINDy.U] = g_of_x[:,2]

        return g

    def u_nominal(
        self, x: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters, using LQR unless
        overridden

        args:
            x: bs x self.n_dims tensor of state
            params: the model parameters used
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        # u = torch.zeros((x.shape[0], self.n_controls))
        
        # Proportional navigation (we don't care about the heading)
        Kp = 10.0
        goal = self.goal_point.squeeze().type_as(x)
        theta_d = torch.atan2(goal[DubinsCarSINDy.Y]- x[:,DubinsCarSINDy.Y], goal[DubinsCarSINDy.X] - x[:,DubinsCarSINDy.X])
        theta_err = theta_d - x[:,DubinsCarSINDy.THETA]
        theta_err = torch.remainder(theta_err + np.pi, 2 * np.pi) - np.pi #wrapToPi(theta_err)
        #u = Kp * theta_err + self.u_eq.type_as(x)

        u = torch.zeros((x.shape[0], self.n_controls))
        u = u.type_as(x)
        u[:,DubinsCarSINDy.U] = Kp * theta_err
        
        return u
    
    def control_affine_ground_truth_dynamics(
        self, x: torch.Tensor, params: Optional[Scenario] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (f, g) representing the system dynamics in control-affine form:

            dx/dt = f(x) + g(x) u

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor representing the control-independent dynamics
            g: bs x self.n_dims x self.n_controls tensor representing the control-
               dependent dynamics
        """
        # Sanity check on input
        assert x.ndim == 2
        assert x.shape[1] == self.n_dims

        # If no params required, use nominal params
        if params is None:
            params = self.nominal_params

        return self._f_ground_truth(x, params), self._g_ground_truth(x, params)
    
    def closed_loop_ground_truth_dynamics(
        self, x: torch.Tensor, u: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        """
        Return the state derivatives at state x and control input u

            dx/dt = f(x) + g(x) u

        args:
            x: bs x self.n_dims tensor of state
            u: bs x self.n_controls tensor of controls
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            xdot: bs x self.n_dims tensor of time derivatives of x
        """
        # Get the control-affine dynamics
        f_ground_truth, g_ground_truth = self.control_affine_ground_truth_dynamics(x, params=params)
        # Compute state derivatives using control-affine form
        xdot = f_ground_truth + torch.bmm(g_ground_truth, u.unsqueeze(-1))

        return xdot.view(x.shape)
    
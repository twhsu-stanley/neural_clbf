"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList, predict_tensor

import pickle

grav = 9.81

# Load the SINDy model ##########################################################################
with open('../pysindy/control_affine_models/saved_models/model_cartpole_sindy', 'rb') as file:
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
print("CP alpha = %4.2f; CP quantile = %5.3f" % (model.model_error['alpha'], cp_quantile))
#################################################################################################

class CartPoleSINDy(ControlAffineSystem):
    """
    Represents SINDy model of a damped inverted pendulum on a cart

    The system has state
        x = [z, theta, v, omega]

    representing the angle and velocity of the pendulum, and it
    has control inputs
        u = [u]

    representing the torque applied.

    The system is parameterized by
        m: pendulum_mass
        M: cart_mass
        L: length
        b: friction
    """

    # Number of states and controls
    N_DIMS = 4
    N_CONTROLS = 1

    # State indices
    Z = 0
    THETA = 1
    V = 2
    OMEGA = 3

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
                            Requires keys ["m", "M", "L", "b"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        self.feature_names = feature_names
        self.coefficients = coefficients
        self.idx_x = idx_x
        self.idx_u = idx_u
        self.cp_quantile = cp_quantile
        
        # TODO: Check if use_linearized_controller = True/False matters
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
        valid = valid and "M" in params
        valid = valid and "m" in params
        valid = valid and "L" in params
        valid = valid and "b" in params

        # Make sure all parameters are physically valid
        valid = valid and params["M"] > 0
        valid = valid and params["m"] > 0
        valid = valid and params["L"] > 0
        valid = valid and params["b"] >= 0

        return valid

    @property
    def n_dims(self) -> int:
        return CartPoleSINDy.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return CartPoleSINDy.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[CartPoleSINDy.Z] = 1.0
        upper_limit[CartPoleSINDy.THETA] = np.pi/6
        upper_limit[CartPoleSINDy.V] = 1.5
        upper_limit[CartPoleSINDy.OMEGA] = 1.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([10.0])
        lower_limit = -torch.tensor([10.0])

        return (upper_limit, lower_limit)
    
    @property
    def goal_point(self):
        return torch.zeros((1, self.n_dims))

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        safe_mask = x.norm(dim=-1) <= 2.0

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        unsafe_mask = x.norm(dim=-1) >= 3.0

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        goal_mask = x.norm(dim=-1) <= 1.0

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

        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # Compute f(x) using the SINDy model
        # TODO
        f_of_x = predict_tensor(x, torch.zeros((batch_size,1)), self.feature_names, self.coefficients, self.idx_x)
        
        f[:, CartPoleSINDy.Z, 0] = f_of_x[:,0]
        f[:, CartPoleSINDy.THETA, 0] = f_of_x[:,1]
        f[:, CartPoleSINDy.V, 0] = f_of_x[:,2]
        f[:, CartPoleSINDy.OMEGA, 0] = f_of_x[:,3]

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
        g[:, CartPoleSINDy.Z, CartPoleSINDy.U] = g_of_x[:,0]
        g[:, CartPoleSINDy.THETA, CartPoleSINDy.U] = g_of_x[:,1]
        g[:, CartPoleSINDy.V, CartPoleSINDy.U] = g_of_x[:,2]
        g[:, CartPoleSINDy.OMEGA, CartPoleSINDy.U] = g_of_x[:,3]

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
        # Compute nominal control from feedback + equilibrium control
        K = self.K.type_as(x)
        goal = self.goal_point.squeeze().type_as(x)
        u_nominal = -(K @ (x - goal).T).T

        # Adjust for the equilibrium setpoint
        u = u_nominal + self.u_eq.type_as(x)

        # Clamp given the control limits
        upper_u_lim, lower_u_lim = self.control_limits
        for dim_idx in range(self.n_controls):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )

        return u
    
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

        # Extract the state variables
        z = x[:, CartPoleSINDy.Z]
        theta = x[:, CartPoleSINDy.THETA]
        v = x[:, CartPoleSINDy.V]
        omega = x[:, CartPoleSINDy.OMEGA]

        # Extract the needed parameters
        M, m = params["M"], params["m"]
        L, b = params["L"], params["b"]
        g = grav

        #################################################################################################
        # TODO:
        F = 0
        det = M + m * torch.mul(torch.sin(theta), torch.sin(theta))
        f_v_dot = (F - b * v - m * L * torch.mul(omega, omega) * torch.sin(theta)  + 0.5 * m * g * torch.sin(2 * theta)) / det
        f_omega_dot = (F * torch.cos(theta) - 0.5 * m * L * torch.mul(omega, omega) * torch.sin(2 * theta) - b * torch.mul(v, torch.cos(theta))
                    + (m + M) * g * torch.sin(theta)) / (det * L)
        #################################################################################################

        f[:, CartPoleSINDy.Z, 0] = v
        f[:, CartPoleSINDy.THETA, 0] = omega
        f[:, CartPoleSINDy.V, 0] = f_v_dot
        f[:, CartPoleSINDy.OMEGA, 0] = f_omega_dot

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
        
        # Extract the needed parameters
        M, m = params["M"], params["m"]
        L, b = params["L"], params["b"]

        # Extract the state variables
        z = x[:, CartPoleSINDy.Z]
        theta = x[:, CartPoleSINDy.THETA]
        v = x[:, CartPoleSINDy.V]
        omega = x[:, CartPoleSINDy.OMEGA]
        
        det = M + m * torch.mul(torch.sin(theta), torch.sin(theta))
        g_v_dot = 1 / det
        g_omega_dot = torch.cos(theta) / (det * L)

        g[:, CartPoleSINDy.Z, CartPoleSINDy.U] = 0
        g[:, CartPoleSINDy.THETA, CartPoleSINDy.U] = 0
        g[:, CartPoleSINDy.V, CartPoleSINDy.U] = g_v_dot
        g[:, CartPoleSINDy.OMEGA, CartPoleSINDy.U] = g_omega_dot

        return g

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

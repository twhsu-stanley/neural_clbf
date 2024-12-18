"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import grav, Scenario, ScenarioList

import pysindy

import dill, pickle

# Load the SINDY model
with open('../pysindy/control_affine_models/saved_models/model_inverted_pendulum_cart_sindy', 'rb') as file:
    model = pickle.load(file)

class InvertedPendulumCartSINDy(ControlAffineSystem):
    """
    Represents SINDy model of a damped inverted pendulum on a cart

    The system has state

        x = [z, z_dot, theta, theta_dot]

    representing the angle and velocity of the pendulum, and it
    has control inputs

        u = [u]

    representing the torque applied.

    The system is parameterized by
        M: mass of cart
        m: mass on the ip
        L: length of the link
        Kd: damping coefficient on the cart
    """

    # Number of states and controls
    N_DIMS = 4
    N_CONTROLS = 1

    # State indices
    Z = 0
    Z_DOT = 1
    THETA = 2
    THETA_DOT = 3

    # Control indices
    U = 0

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.00,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
    ):
        """
        Initialize the inverted pendulum.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["m", "L", "b"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """

        # SINDY model
        #self.model = learned_model

        # Get indices of the SINDy regressor corresponding to each state and control input
        feature_names = model.get_feature_names()
        idx_x = [] # Indices for f(x)
        idx_u = [] # Indices for g(x)*u
        for i in range(len(feature_names)):
            if 'u0' in feature_names[i]:
                idx_u.append(i)
            else:
                idx_x.append(i)
        self.idx_x = idx_x
        self.idx_u = idx_u
        
        # TODO: Check if use_linearized_controller = True/False matters
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios
        )

        # Since we aren't using a linearized controller, we need to provide
        # some guess for a Lyapunov matrix
        #self.P = torch.eye(self.n_dims)
        #self.P = torch.tensor([[ 5.6748,  3.2077, -8.2618, -1.8670],
        #                       [ 3.2077,  2.4812, -6.6064, -1.4238],
        #                       [-8.2618, -6.6064, 19.8187,  3.8687],
        #                       [-1.8670, -1.4238,  3.8687,  0.8558]])

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
        valid = valid and "Kd" in params

        # Make sure all parameters are physically valid
        valid = valid and params["M"] > 0
        valid = valid and params["m"] > 0
        valid = valid and params["L"] > 0
        valid = valid and params["Kd"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return InvertedPendulumCartSINDy.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        #return [InvertedPendulumCartSINDy.THETA]
        return [] # Testing

    @property
    def n_controls(self) -> int:
        return InvertedPendulumCartSINDy.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[InvertedPendulumCartSINDy.Z] = 8.0
        upper_limit[InvertedPendulumCartSINDy.Z_DOT] = 10.0
        upper_limit[InvertedPendulumCartSINDy.THETA] = np.pi/2
        upper_limit[InvertedPendulumCartSINDy.THETA_DOT] = 10.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([100.0])
        lower_limit = -torch.tensor([100.0])

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
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # Compute f(x) using the SINDy model
        Theta = model.get_regressor(x.detach().numpy(), u = np.ones((batch_size,1)))
        coeff = model.optimizer.coef_
        Theta_x = Theta[:,self.idx_x]
        coeff_x = coeff[:,self.idx_x]
        f_of_x = Theta_x @ coeff_x.T

        # Convert AxesArray to tensor
        f_of_x = torch.tensor(f_of_x)

        # ISSUE: Converting x to a numpy array and then converting it back to a tensor is causing the loss of gradient,
        #        making the Jacobian (needed for linearizing the model and obtaining the LQR gain) always zero. 
        # TODO: The best solution seems to be making the in/output of model.get_regressor() both tensors

        #z = x[:, InvertedPendulumCartSINDy.Z]
        #z_dot = x[:, InvertedPendulumCartSINDy.Z_DOT]
        #theta = x[:, InvertedPendulumCartSINDy.THETA]
        #theta_dot = x[:, InvertedPendulumCartSINDy.THETA_DOT]

        f[:, InvertedPendulumCartSINDy.Z, 0] = f_of_x[:,0]
        f[:, InvertedPendulumCartSINDy.Z_DOT, 0] = f_of_x[:,1]
        f[:, InvertedPendulumCartSINDy.THETA, 0] = f_of_x[:,2]
        f[:, InvertedPendulumCartSINDy.THETA_DOT, 0] = f_of_x[:,3]

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
        Theta = model.get_regressor(x.detach().numpy(), u = np.ones((batch_size,1)))
        coeff = model.optimizer.coef_
        Theta_u = Theta[:,self.idx_u]
        coeff_u = coeff[:,self.idx_u]
        g_of_x = Theta_u @ coeff_u.T

        # Convert AxesArray to tensor
        g_of_x = torch.tensor(g_of_x)

        # Effect on theta dot
        g[:, InvertedPendulumCartSINDy.Z, InvertedPendulumCartSINDy.U] = g_of_x[:,0]
        g[:, InvertedPendulumCartSINDy.Z_DOT, InvertedPendulumCartSINDy.U] = g_of_x[:,1]
        g[:, InvertedPendulumCartSINDy.THETA, InvertedPendulumCartSINDy.U] = g_of_x[:,2]
        g[:, InvertedPendulumCartSINDy.THETA_DOT, InvertedPendulumCartSINDy.U] = g_of_x[:,3]

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
        K = self.K.type_as(x) # torch.tensor([[-0.9500, -20.1210, 77.3411, 15.1780]])
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

        # Extract the needed parameters
        M, m = params["M"], params["m"]
        L, Kd = params["L"], params["Kd"]

        # Extract the state variables
        z = x[:, InvertedPendulumCartSINDy.Z]
        z_dot = x[:, InvertedPendulumCartSINDy.Z_DOT]
        theta = x[:, InvertedPendulumCartSINDy.THETA]
        theta_dot = x[:, InvertedPendulumCartSINDy.THETA_DOT]

        f_z_ddot = (-Kd*z_dot - m*L*theta_dot**2*torch.sin(theta) + m*grav*torch.sin(theta)*torch.cos(theta)) / (M + m*torch.sin(theta)**2)
        f_theta_ddot = (grav*torch.sin(theta) + (-Kd*z_dot - m*L*theta_dot**2*torch.sin(theta))*torch.cos(theta)/(M + m)) / (L - m*L*torch.cos(theta)**2/(M + m))

        f[:, InvertedPendulumCartSINDy.Z, 0] = z_dot
        f[:, InvertedPendulumCartSINDy.Z_DOT, 0] = f_z_ddot
        f[:, InvertedPendulumCartSINDy.THETA, 0] = theta_dot
        f[:, InvertedPendulumCartSINDy.THETA_DOT, 0] = f_theta_ddot

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
        L, Kd = params["L"], params["Kd"]

        # Extract the state variables
        #z = x[:, InvertedPendulumCartSINDy.Z]
        #z_dot = x[:, InvertedPendulumCartSINDy.Z_DOT]
        theta = x[:, InvertedPendulumCartSINDy.THETA]
        #theta_dot = x[:, InvertedPendulumCartSINDy.THETA_DOT]
        
        g_z_ddot = 1 / (M + m*torch.sin(theta)**2)
        g_theta_ddot = (torch.cos(theta)/(M + m)) / (L - m*L*torch.cos(theta)**2/(M + m))

        #g[:, InvertedPendulumCartSINDy.Z, InvertedPendulumCartSINDy.U] = 0
        g[:, InvertedPendulumCartSINDy.Z_DOT, InvertedPendulumCartSINDy.U] = g_z_ddot
        #g[:, InvertedPendulumCartSINDy.THETA, InvertedPendulumCartSINDy.U] = 0
        g[:, InvertedPendulumCartSINDy.THETA_DOT, InvertedPendulumCartSINDy.U] = g_theta_ddot

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

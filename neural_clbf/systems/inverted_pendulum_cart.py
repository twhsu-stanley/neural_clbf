"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import grav, Scenario, ScenarioList

class InvertedPendulumCart(ControlAffineSystem):
    """
    Represents a inverted pendulum on a cart

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
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
    ):
        """
        Initialize the inverted pendulum.

        args:
            nominal_params: a dictionary giving the parameter values for the system
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
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
        valid = valid and "Kd" in params

        # Make sure all parameters are physically valid
        valid = valid and params["M"] > 0
        valid = valid and params["m"] > 0
        valid = valid and params["L"] > 0
        valid = valid and params["Kd"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return InvertedPendulumCart.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        #return [InvertedPendulumCart.THETA]
        return [] # Testing

    @property
    def n_controls(self) -> int:
        return InvertedPendulumCart.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[InvertedPendulumCart.Z] = 8.0
        upper_limit[InvertedPendulumCart.Z_DOT] = 10.0
        upper_limit[InvertedPendulumCart.THETA] = np.pi/2
        upper_limit[InvertedPendulumCart.THETA_DOT] = 10.0

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

        # Extract the needed parameters
        M, m = params["M"], params["m"]
        L, Kd = params["L"], params["Kd"]

        # Extract the state variables
        z = x[:, InvertedPendulumCart.Z]
        z_dot = x[:, InvertedPendulumCart.Z_DOT]
        theta = x[:, InvertedPendulumCart.THETA]
        theta_dot = x[:, InvertedPendulumCart.THETA_DOT]

        f_z_ddot = (-Kd*z_dot - m*L*theta_dot**2*torch.sin(theta) + m*grav*torch.sin(theta)*torch.cos(theta)) / (M + m*torch.sin(theta)**2)
        f_theta_ddot = (grav*torch.sin(theta) + (-Kd*z_dot - m*L*theta_dot**2*torch.sin(theta))*torch.cos(theta)/(M + m)) / (L - m*L*torch.cos(theta)**2/(M + m))

        f[:, InvertedPendulumCart.Z, 0] = z_dot
        f[:, InvertedPendulumCart.Z_DOT, 0] = f_z_ddot
        f[:, InvertedPendulumCart.THETA, 0] = theta_dot
        f[:, InvertedPendulumCart.THETA_DOT, 0] = f_theta_ddot

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
        
        # Extract the needed parameters
        M, m = params["M"], params["m"]
        L, Kd = params["L"], params["Kd"]

        # Extract the state variables
        #z = x[:, InvertedPendulumCart.Z]
        #z_dot = x[:, InvertedPendulumCart.Z_DOT]
        theta = x[:, InvertedPendulumCart.THETA]
        #theta_dot = x[:, InvertedPendulumCart.THETA_DOT]
        
        g_z_ddot = 1 / (M + m*torch.sin(theta)**2)
        g_theta_ddot = (torch.cos(theta)/(M + m)) / (L - m*L*torch.cos(theta)**2/(M + m))

        #g[:, InvertedPendulumCart.Z, InvertedPendulumCart.U] = 0
        g[:, InvertedPendulumCart.Z_DOT, InvertedPendulumCart.U] = g_z_ddot
        #g[:, InvertedPendulumCart.THETA, InvertedPendulumCart.U] = 0
        g[:, InvertedPendulumCart.THETA_DOT, InvertedPendulumCart.U] = g_theta_ddot

        return g
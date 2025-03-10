"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List
from math import sqrt

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class DubinsCar(ControlAffineSystem):
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
        return DubinsCar.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []#[2]

    @property
    def n_controls(self) -> int:
        return DubinsCar.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[DubinsCar.X] = 10.0
        upper_limit[DubinsCar.Y] = 7.0
        upper_limit[DubinsCar.THETA] = np.pi

        lower_limit = torch.ones(self.n_dims)
        lower_limit[DubinsCar.X] = 2.0
        lower_limit[DubinsCar.Y] = 1.0
        lower_limit[DubinsCar.THETA] = -np.pi

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
        return torch.tensor([[2.0, 4.0, 0.0]])
    
    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        d = torch.pow(x[:,DubinsCar.X] - 5.0, 2) + torch.pow(x[:,DubinsCar.Y] - 4.0, 2) - 4
        v = self.nominal_params['v']
        drate =  (2 * (x[:,DubinsCar.X] - 5.0) * v * torch.cos(x[:,DubinsCar.THETA]) + 2 * (x[:,DubinsCar.Y] - 4.0) * v * torch.sin(x[:,DubinsCar.THETA]))
        h = 15 * d + drate
        safe_mask.logical_and_(h >= 33.75)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        d = torch.pow(x[:,DubinsCar.X] - 5.0, 2) + torch.pow(x[:,DubinsCar.Y] - 4.0, 2) - 4
        v = self.nominal_params['v']
        drate =  (2 * (x[:,DubinsCar.X] - 5.0) * v * torch.cos(x[:,DubinsCar.THETA]) + 2 * (x[:,DubinsCar.Y] - 4.0) * v * torch.sin(x[:,DubinsCar.THETA]))
        h = 15 * d + drate
        unsafe_mask.logical_or_(h < 0)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of points in the state space
        """
        # TODO: verify if the goal_mask is not used at all
        goal_mask = (torch.pow(x[:,DubinsCar.X] - 2.0, 2) + torch.pow(x[:,DubinsCar.Y] - 4.0, 2) <= 0.1)

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
        v = params["v"]
        
        # and state variables
        #x_ = x[:, DubinsCar.X]
        #y_ = x[:, DubinsCar.Y]
        theta_ = x[:, DubinsCar.THETA]

        f[:, DubinsCar.X, 0] = v * torch.cos(theta_)
        f[:, DubinsCar.Y, 0] = v * torch.sin(theta_)
        f[:, DubinsCar.THETA, 0] = 0.0

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

        # The control inputs are accelerations
        g[:, DubinsCar.X, DubinsCar.U] = 0.0
        g[:, DubinsCar.Y, DubinsCar.U] = 0.0
        g[:, DubinsCar.THETA, DubinsCar.U] = 1.0

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
        theta_d = torch.atan2(goal[DubinsCar.Y]- x[:,DubinsCar.Y], goal[DubinsCar.X] - x[:,DubinsCar.X])
        theta_err = theta_d - x[:,DubinsCar.THETA]
        theta_err = torch.remainder(theta_err + np.pi, 2 * np.pi) - np.pi #wrapToPi(theta_err)
        #u = Kp * theta_err + self.u_eq.type_as(x)

        u = torch.zeros((x.shape[0], self.n_controls))
        u = u.type_as(x)
        u[:,DubinsCar.U] = Kp * theta_err
        
        return u
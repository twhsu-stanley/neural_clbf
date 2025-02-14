"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List
from math import sqrt

import torch

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class ACC(ControlAffineSystem):
    """
    Adaptive cruise control
    """

    # Number of states and controls
    N_DIMS = 3
    N_CONTROLS = 1

    # State indices
    P = 0
    V = 1
    Z = 2

    # Control indices
    U = 0

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
        valid = valid and "m" in params
        valid = valid and "f0" in params
        valid = valid and "f1" in params
        valid = valid and "f2" in params
        valid = valid and "v0" in params
        valid = valid and "vd" in params
        valid = valid and "Th" in params

        # Make sure all parameters are physically valid
        valid = valid and params["m"] > 0
        valid = valid and params["v0"] > 0
        valid = valid and params["vd"] > 0
        valid = valid and params["Th"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return ACC.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return ACC.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[ACC.P] = 100.0
        upper_limit[ACC.V] = 35.0
        upper_limit[ACC.Z] = 30.0

        lower_limit = torch.ones(self.n_dims)
        lower_limit[ACC.P] = 0.0
        lower_limit[ACC.V] = 0.0
        lower_limit[ACC.Z] = 0.0

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([30000.0])
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)
        Th = self.nominal_params['Th']
        safe_mask.logical_and_(x[:, ACC.Z] - Th * x[:, ACC.V] >= 0.1)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)
        Th = self.nominal_params['Th']
        unsafe_mask.logical_or_(x[:, ACC.Z] - Th * x[:, ACC.V] <= 0.0)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of points in the state space
        """
        Th = self.nominal_params['Th']
        goal_mask = x[:, ACC.Z] - Th * x[:, ACC.V] >= 0.1

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

        f0 = params["f0"]
        f1 = params["f1"]
        f2 = params["f2"]
        m = params["m"]
        v0 = params["v0"]

        v_ = x[:, ACC.V]
        #z_ = x[:, ACC.Z]

        Fr = f0 + f1 * v_ + f2 * torch.pow(v_, 2)

        # The first three dimensions just integrate the velocity
        f[:, ACC.P, 0] = v_
        f[:, ACC.V, 0] = -Fr/m
        f[:, ACC.Z, 0] = v0 - v_

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

        m = params["m"]

        g[:, ACC.P, ACC.U] = 0.0
        g[:, ACC.V, ACC.U] = 1/m
        g[:, ACC.Z, ACC.U] = 0.0

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
        vd = params["vd"]
        Kp = 100.0
        u = torch.zeros((x.shape[0], self.n_controls))
        u = u.type_as(x)
        u[:,ACC.U] = Kp * (vd - x[:,ACC.V])
        
        return u
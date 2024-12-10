"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import grav, Scenario, ScenarioList

class DoubleInvertedPendulum(ControlAffineSystem):
    """
    Represents a double inverted pendulum on a cart

    The system has state

        x = [q, q_dot, theta_1, theta_1_dot, theta_2, theta_2_dot]

    representing the angle and velocity of the pendulum, and it
    has control inputs

        u = [u]

    representing the torque applied.

    The system is parameterized by
        M: mass of cart
        m1: mass on the first link
        m2: mass on the second link
        L1: length of the first link
        L2: length of the second link
    """

    # Number of states and controls
    N_DIMS = 6
    N_CONTROLS = 1

    # State indices
    X = 0
    X_DOT = 1
    THETA_1 = 2
    THETA_1_DOT = 3
    THETA_2 = 4
    THETA_2_DOT = 5

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
                            Requires keys ["m", "L", "b"]
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
        valid = valid and "m1" in params
        valid = valid and "m2" in params
        valid = valid and "L1" in params
        valid = valid and "L2" in params

        # Make sure all parameters are physically valid
        valid = valid and params["M"] > 0
        valid = valid and params["m1"] > 0
        valid = valid and params["m2"] > 0
        valid = valid and params["L1"] > 0
        valid = valid and params["L2"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return DoubleInvertedPendulum.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [DoubleInvertedPendulum.THETA_1, DoubleInvertedPendulum.THETA_2]
        #return [] # Testing

    @property
    def n_controls(self) -> int:
        return DoubleInvertedPendulum.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[DoubleInvertedPendulum.X] = 10.0
        upper_limit[DoubleInvertedPendulum.X_DOT] = 20.0
        upper_limit[DoubleInvertedPendulum.THETA_1] = np.pi
        upper_limit[DoubleInvertedPendulum.THETA_1_DOT] = np.pi * 15
        upper_limit[DoubleInvertedPendulum.THETA_2] = np.pi
        upper_limit[DoubleInvertedPendulum.THETA_2_DOT] = np.pi * 15

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([1000.0])
        lower_limit = -torch.tensor([1000.0])

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
        safe_mask = x.norm(dim=-1) <= 5.0

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        unsafe_mask = x.norm(dim=-1) >= 8.0

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        goal_mask = x.norm(dim=-1) <= 2.0

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
        A = torch.zeros((batch_size, 3, 3))
        A = A.type_as(x)

        # Extract the needed parameters
        M, m1, m2 = params["M"], params["m1"], params["m2"]
        L1, L2 = params["L1"], params["L2"]
        
        # Extract the state variables
        q = x[:, DoubleInvertedPendulum.X]
        q_dot = x[:, DoubleInvertedPendulum.X_DOT]
        theta_1 = x[:, DoubleInvertedPendulum.THETA_1]
        theta_1_dot = x[:, DoubleInvertedPendulum.THETA_1_DOT]
        theta_2 = x[:, DoubleInvertedPendulum.THETA_2]
        theta_2_dot = x[:, DoubleInvertedPendulum.THETA_2_DOT] 

        # A(state) * state_ddot = b(state, state_dot) + dL_dstate + u
        # => A(state) * state_ddot = c(state, state_dot, u)
        # => state_ddot = inv( A(state) ) * c(state, state_dot, u)
        dL_dx = 0.0
        dL_da = -(m1 + m2) * L1 * theta_1_dot * q_dot * torch.sin(theta_1) + (m1 + m2) * grav * L1 * torch.sin(theta_1) - m2 * L1 * L2 * theta_1_dot * theta_2_dot * torch.sin(theta_1 - theta_2)
        dL_db = m2 * L2 * (grav * torch.sin(theta_2) + L1 * theta_1_dot * theta_2_dot * torch.sin(theta_1 - theta_2) - q_dot * theta_2_dot * torch.sin(theta_2))

        a11 = M + m1 + m2
        a12 = (m1 + m2) * L1 * torch.cos(theta_1)
        a13 = m2 * L2 * torch.cos(theta_2)
        b1 = (m1 + m2) * L1 * theta_1_dot ** 2 * torch.sin(theta_1) + m2 * L2 * theta_2_dot ** 2 * torch.sin(theta_2)

        a21 = (m1 + m2) * L1 * torch.cos(theta_1)
        a22 = (m1 + m2) * L1 ** 2
        a23 = m2 * L1 * L2 * torch.cos(theta_1 - theta_2)
        b2 = (m1 + m2) * q_dot * theta_1_dot * L1 * torch.sin(theta_1) + m2 * L1 * L2 * theta_2_dot * (theta_1_dot - theta_2_dot) * torch.sin(theta_1 - theta_2)

        a31 = m2 * L2 * torch.cos(theta_2)
        a32 = m2 * L1 * L2 * torch.cos(theta_1 - theta_2)
        a33 = m2 * L2 ** 2
        b3 = m2 * q_dot * theta_2_dot * L2 * torch.sin(theta_2) + m2 * L1 * L2 * theta_1_dot * (theta_1_dot - theta_2_dot) * torch.sin(theta_1 - theta_2)

        A[:,0,0] = a11
        A[:,0,1] = a12
        A[:,0,2] = a13
        A[:,1,0] = a21 
        A[:,1,1] = a22
        A[:,1,2] = a23
        A[:,2,0] = a31
        A[:,2,1] = a32
        A[:,2,2] = a33

        c = torch.stack([b1 + dL_dx, b2 + dL_da, b3 + dL_db], dim = 1)

        #f_of_x = A_inv @ c
        f_of_x = torch.linalg.solve(A, c)

        assert not torch.isnan(f_of_x).any()
        assert not torch.isinf(f_of_x).any()

        q_ddot = f_of_x[:,0]
        theta_1_ddot = f_of_x[:,1]
        theta_2_ddot = f_of_x[:,2]

        f[:, DoubleInvertedPendulum.X, 0] = q_dot
        f[:, DoubleInvertedPendulum.THETA_1, 0] = theta_1_dot
        f[:, DoubleInvertedPendulum.THETA_2, 0] = theta_2_dot

        f[:, DoubleInvertedPendulum.X_DOT, 0] = q_ddot
        f[:, DoubleInvertedPendulum.THETA_1_DOT, 0] = theta_1_ddot
        f[:, DoubleInvertedPendulum.THETA_2_DOT, 0] = theta_2_ddot

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
        A = torch.zeros((batch_size, 3, 3))
        A = A.type_as(x)

        # Extract the needed parameters
        M, m1, m2 = params["M"], params["m1"], params["m2"]
        L1, L2 = params["L1"], params["L2"]

        # Extract the state variables
        q = x[:, DoubleInvertedPendulum.X]
        #q_dot = x[:, DoubleInvertedPendulum.X_DOT]
        theta_1 = x[:, DoubleInvertedPendulum.THETA_1]
        #theta_1_dot = x[:, DoubleInvertedPendulum.THETA_1_DOT]
        theta_2 = x[:, DoubleInvertedPendulum.THETA_2]
        #theta_2_dot = x[:, DoubleInvertedPendulum.THETA_2_DOT] 
        
        a11 = M + m1 + m2
        a12 = (m1 + m2) * L1 * torch.cos(theta_1)
        a13 = m2 * L2 * torch.cos(theta_2)

        a21 = (m1 + m2) * L1 * torch.cos(theta_1)
        a22 = (m1 + m2) * L1 ** 2
        a23 = m2 * L1 * L2 * torch.cos(theta_1 - theta_2)
        
        a31 = m2 * L2 * torch.cos(theta_2)
        a32 = m2 * L1 * L2 * torch.cos(theta_1 - theta_2)
        a33 = m2 * L2 ** 2
        
        A[:,0,0] = a11
        A[:,0,1] = a12
        A[:,0,2] = a13
        A[:,1,0] = a21
        A[:,1,1] = a22
        A[:,1,2] = a23
        A[:,2,0] = a31
        A[:,2,1] = a32
        A[:,2,2] = a33

        #A_inv = torch.linalg.inv(A)
        #g_of_x = A_inv @ torch.tensor([1.0, 0.0, 0.0]) # inv(A) @ [1;0;0]
        g_of_x = torch.linalg.solve(A, torch.tensor([1.0, 0.0, 0.0]))

        assert not torch.isnan(g_of_x).any()
        assert not torch.isinf(g_of_x).any()

        #g[:, DoubleInvertedPendulum.X, DoubleInvertedPendulum.U] = 0
        g[:, DoubleInvertedPendulum.X_DOT, DoubleInvertedPendulum.U] = g_of_x[:,0]
        #g[:, DoubleInvertedPendulum.THETA_1, DoubleInvertedPendulum.U] = 0
        g[:, DoubleInvertedPendulum.THETA_1_DOT, DoubleInvertedPendulum.U] = g_of_x[:,1]
        #g[:, DoubleInvertedPendulum.THETA_2, DoubleInvertedPendulum.U] = 0
        g[:, DoubleInvertedPendulum.THETA_2_DOT, DoubleInvertedPendulum.U] = g_of_x[:,2]

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
        K = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        goal = self.goal_point.squeeze().type_as(x)
        u_nominal = -(K @ (x - goal).T).T

        # Adjust for the equilibrium setpoint
        u = u_nominal + self.u_eq.type_as(x)

        # Clamp given the control limits
        #upper_u_lim, lower_u_lim = self.control_limits
        #for dim_idx in range(self.n_controls):
        #    u[:, dim_idx] = torch.clamp(
        #        u[:, dim_idx],
        #        min=lower_u_lim[dim_idx].item(),
        #        max=upper_u_lim[dim_idx].item(),
        #    )

        return u
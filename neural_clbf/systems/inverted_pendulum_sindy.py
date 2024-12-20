"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import grav, Scenario, ScenarioList

import pysindy as ps
from pysindy import SINDy

class InvertedPendulumSINDy(ControlAffineSystem):
    """
    Represents a damped inverted pendulum.

    The system has state

        x = [theta, theta_dot]

    representing the angle and velocity of the pendulum, and it
    has control inputs

        u = [u]

    representing the torque applied.

    The system is parameterized by
        m: mass
        L: length of the pole
        b: damping
    """

    # Number of states and controls
    N_DIMS = 2
    N_CONTROLS = 1

    # State indices
    THETA = 0
    THETA_DOT = 1
    # Control indices
    U = 0

    def __init__(
        self,
        nominal_params: Scenario,
        learned_model: SINDy,
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

        # SINDY model
        self.model = learned_model

        # Get indices of the SINDy regressor corresponding to each state and control input
        feature_names = self.model.get_feature_names()
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
            nominal_params, dt=dt, controller_dt=controller_dt, use_linearized_controller=False, scenarios=scenarios
        )

        # Since we aren't using a linearized controller, we need to provide
        # some guess for a Lyapunov matrix
        self.P = torch.eye(self.n_dims)

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["m", "L", "b"]
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True

        return valid

    @property
    def n_dims(self) -> int:
        return InvertedPendulumSINDy.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [InvertedPendulumSINDy.THETA]

    @property
    def n_controls(self) -> int:
        return InvertedPendulumSINDy.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[InvertedPendulumSINDy.THETA] = 2.0
        upper_limit[InvertedPendulumSINDy.THETA_DOT] = 2.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([100 * 10.0])
        lower_limit = -torch.tensor([100 * 10.0])

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        safe_mask = x.norm(dim=-1) <= 0.5

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        unsafe_mask = x.norm(dim=-1) >= 1.5

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        goal_mask = x.norm(dim=-1) <= 0.3

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
        Theta = self.model.get_regressor(x.detach().numpy(), u = np.ones((batch_size,1)))
        coeff = self.model.optimizer.coef_
        Theta_x = Theta[:,self.idx_x]
        coeff_x = coeff[:,self.idx_x]
        f_of_x = Theta_x @ coeff_x.T

        # Convert AxesArray to tensor
        f_of_x = torch.tensor(f_of_x)

        # ISSUE: Converting x to a numpy array and then converting it back to a tensor is causing the loss of gradient,
        #        making the Jacobian (needed for linearizing the model and obtaining the LQR gain) always zero. 
        # TODO: The best solution seems to be making the in/output of model.get_regressor() both tensors

        f[:, InvertedPendulumSINDy.THETA, 0] = f_of_x[:,0]
        f[:, InvertedPendulumSINDy.THETA_DOT, 0] = f_of_x[:,1]

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
        Theta = self.model.get_regressor(x.detach().numpy(), u = np.ones((batch_size,1)))
        coeff = self.model.optimizer.coef_
        Theta_u = Theta[:,self.idx_u]
        coeff_u = coeff[:,self.idx_u]
        g_of_x = Theta_u @ coeff_u.T

        # Convert AxesArray to tensor
        g_of_x = torch.tensor(g_of_x)

        # Effect on theta dot
        g[:, InvertedPendulumSINDy.THETA_DOT, InvertedPendulumSINDy.U] = g_of_x[:,1]

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
        K = torch.tensor([[18.3128,  5.8956]])
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
        m, L, b = params["m"], params["L"], params["b"]
        # and state variables
        theta = x[:, InvertedPendulumSINDy.THETA]
        theta_dot = x[:, InvertedPendulumSINDy.THETA_DOT]

        # The derivatives of theta is just its velocity
        f[:, InvertedPendulumSINDy.THETA, 0] = theta_dot

        # Acceleration in theta depends on theta via gravity and theta_dot via damping
        f[:, InvertedPendulumSINDy.THETA_DOT, 0] = (
            grav / L * torch.sin(theta) - b / (m * L ** 2) * theta_dot
        )

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
        m, L = params["m"], params["L"]

        # Effect on theta dot
        g[:, InvertedPendulumSINDy.THETA_DOT, InvertedPendulumSINDy.U] = 1 / (m * L ** 2)

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

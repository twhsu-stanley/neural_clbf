"""Defines useful constants and helper functions for dynamical systems"""
from typing import Dict, List

import numpy as np
import scipy.linalg
import cvxpy as cp
import torch
torch.set_default_dtype(torch.float64)

# Gravitation acceleration
grav = 9.80665

# Define a type alias for parameter scenarios
Scenario = Dict[str, float]
ScenarioList = List[Scenario]

def lqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    return_eigs: bool = False,
):
    """Solve the discrete time lqr controller.

    x_{t+1} = A x_t + B u_t

    cost = sum x.T*Q*x + u.T*R*u

    Code adapted from Mark Wilfred Mueller's continuous LQR code at
    http://www.mwm.im/lqr-controllers-with-python/

    Based on Bertsekas, p.151

    Yields the control law u = -K x
    """

    # first, try to solve the ricatti equation
    X = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = scipy.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    if not return_eigs:
        return K
    else:
        eigVals, _ = scipy.linalg.eig(A - B * K)
        return K, eigVals


def continuous_lyap(Acl: np.ndarray, Q: np.ndarray):
    """Solve the continuous time lyapunov equation.

    Acl.T P + P Acl + Q = 0

    using scipy, which expects AP + PA.T = Q, so we need to transpose Acl and negate Q
    """
    P = scipy.linalg.solve_continuous_lyapunov(Acl.T, -Q)
    return P


def discrete_lyap(Acl: np.ndarray, Q: np.ndarray):
    """Solve the continuous time lyapunov equation.

    Acl.T P Acl - P + Q = 0

    using scipy, which expects A P A.T - P + Q = 0, so we need to transpose Acl
    """
    P = scipy.linalg.solve_discrete_lyapunov(Acl.T, Q)
    return P


def robust_continuous_lyap(Acl_list: List[np.ndarray], Q: np.ndarray):
    """Solve the continuous time lyapunov equation robustly. That is, find P such that

    Acl.T P + P Acl <= -Q

    for each A
    """
    # Sanity check the provided scenarios. They should all have the same dimension
    # and they should all be stable
    n_dims = Q.shape[0]
    for Acl in Acl_list:
        assert Acl.shape == Q.shape, "Acl shape should be consistent with Q"
        assert (np.linalg.eigvals(Acl) < 0).all(), "Acl should be stable"

    # We'll find P using a semidefinite program. First we need a matrix variable for P
    P = cp.Variable((n_dims, n_dims), symmetric=True)

    # Each scenario implies a semidefiniteness constraint
    constraints = [P >> 0.1 * Q]  # P must itself be semidefinite
    for Acl in Acl_list:
        constraints.append(Acl.T @ P + P @ Acl << -P)

    # The objective is to minimize the size of the elements of P
    objective = cp.trace(np.ones((n_dims, n_dims)) @ P)

    # Solve!
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()

    return P.value

def jacobian_numpy(f, x0, delta = 1e-5) -> np.ndarray:
    """
    Compute the Jacobian matrix of a vector-valued function 'f' at point 'x0' using numpy
    """
    assert f(x0).shape[0] == x0.shape[1]

    # Convert x0 to a numpy array
    #x0 = x0.cpu().detach().numpy()
    
    # Evaluate the function at x0
    f0 = f(x0)
    
    # Initialize the Jacobian matrix
    J = np.zeros((f(x0).shape[0], x0.shape[1]))
    
    for i in range(f(x0).shape[0]):
        x_perturbed = x0.detach().clone()
        for j in range(x0.shape[1]):
            x_perturbed[0][j] += delta
            f_perturbed = f(x_perturbed)
            J[i, j] = (f_perturbed[i] - f0[i]) / delta
            x_perturbed[0][j] = x0[0][j]  # Reset the perturbed value
    
    return J

def predict_tensor(x, u, feature_names, coefficients, feature_indices = None):
    """ Compute the model predciction using expressions for torch tensor operations"""
    # x: array or tensor (batch_size x n_states)
    # u: array or tensor (batch_size x n_controls)
    # feature_names: list (len = n_features)
    # coefficients: array (size = n_states x n_features)

    if feature_indices is None:
        n_features = len(feature_names)
        feature_indices = range(n_features)
    
    n_states = coefficients.shape[0]
    n_controls = u.shape[1]
    batch_size = x.shape[0]

    for s in range(n_states):
        locals()[f'x{s}'] = x[:,s]

    for s in range(n_controls):
        locals()[f'u{s}'] = u[:,s]

    f = torch.zeros((batch_size, n_states), dtype = torch.float64)
    for s in range(n_states):
        for i in feature_indices:
            f[:,s] += eval(feature_names[i]) * coefficients[s,i]
    return f

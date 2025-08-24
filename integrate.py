from typing import Callable
import numpy as np
from potentials import Potential, PowerPotential
import abc

class DerivativeBase(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: float, y: float) -> float:
        pass

class DerivativeFromFunction(DerivativeBase):
    def __init__(self, func: Callable[[float, float], float]):
        self.func = func

    def __call__(self, x: float, y: float) -> float:
        return self.func(x, y)

class HJDerivative(DerivativeBase):
    """Hamilton-Jacobi derivative for inflationary dynamics."""
    def __init__(self, V: Potential = PowerPotential()):
        self.V = V

    def __call__(self, phi: float, H: float) -> float:
        # Hamilton-Jacobi equation: H' = sqrt(1.5*H^2 - 0.5*V(φ))
        return grad_calc(H, phi, self.V)

def grad_calc(H: float, phi: float, V: Potential = PowerPotential()):

    """Calculate the magnitude of the gradient dH/dφ (used for RK4 steps)."""
    # 1.5*H^2 - 0.5*V0*phi^m corresponds to the sign-adjusted derivative (squared) of H.
    # We take the square root of its absolute value for the magnitude.
    return np.sqrt(abs(grad_sqr(H, phi, V)))

def grad_sqr(H: float, phi: float, V: Potential = PowerPotential()):

    """Calculate the squared gradient 1.5*H^2 - 0.5*V(φ) (includes sign for H')."""
    return 1.5 * (H ** 2) - 0.5 * V(phi)

def epsilon_calc(H: float, phi: float, V: Potential = PowerPotential()):
    """Calculate the slow-roll parameter ε = 2|H'|/H (squared form used for efficiency)."""
    # ε = 2 * |(dH/dφ)| / H, where (dH/dφ)^2 = 1.5*H^2 - 0.5*V(φ) (Hamilton-Jacobi equation)
    return 2 * abs(1.5 * (H ** 2) - 0.5 * V(phi)) / (H ** 2)

def rk4_step(x: float, y: float, step: float = -0.05, derivative: DerivativeBase = HJDerivative()):
    assert step != 0, "Input step size must not be zero for RK4 step."
    # Perform one 4th-order Runge-Kutta step
    dx = step
    k1 = derivative(x, y)
    k2 = derivative(x + 0.5 * dx * k1, y + 0.5 * dx)
    k3 = derivative(x + 0.5 * dx * k2, y + 0.5 * dx)
    k4 = derivative(x + dx * k3, y + dx)
    # Update the fields
    x += dx
    y += (1/6) * (k1 + 2*k2 + 2*k3 + k4) * dx
    return x, y

def integrand_for_N(ε: float) -> float:
    """Integrand for the number of e-folds N."""
    return 1 / (2 * ε)**0.5

g = lambda x: x if x>0.00001 else 0.00001
def f(H: float, phi: float, V: Potential = PowerPotential()):
    '''Interpolation of step-size so that particularly small ε values have small enough step sizes. 
    This could be particularly helpful with ϕ^m potentials for small m>0'''
    returnval = (0.5**((0.0001/g(epsilon_calc(H, phi, V=V))) if epsilon_calc(H, phi, V=V) < 0.0001 else 0.5))
    return returnval

def adapt_step_factor(H: float, phi: float, dphi: float, V: Potential = PowerPotential()):
    """Adapt the step size based on the current value of ε."""
    coeff = f(H, phi, V=V)
    if grad_sqr(H, phi, V=V) == 0:
        raise ValueError("Gradient squared is zero, cannot adapt step size.\nH: {}, φ: {}".format(H, phi))
    return min(coeff*(abs(H - (V(phi)/3)**0.5))/grad_calc(H, phi, V=V), abs(dphi))

def get_trajectory_decreasing_phi(V: Potential, phi_0: float, H_0: float, dphi_0 = -0.0001, bound_multiplier=100, suppress=False, track_N_time=True):
    """Get the trajectory of H against φ for a decreasing φ field under the potential V(φ)."""

    # We check whether the initial point is in the right domain
    if not (3 * (H_0 ** 2) > V(phi_0) > 2 * (H_0 ** 2) or suppress):
        raise ValueError("Initial conditions are not feasible")

    # Initialize variables
    max_n = bound_multiplier * int(abs(phi_0 / dphi_0))
    phi_arr = np.zeros(max_n + 1)
    H_arr = np.zeros_like(phi_arr)
    phi_arr[0] = phi_0
    H_arr[0] = H_0
    dphi = -adapt_step_factor(H_0, phi_0, dphi_0, V=V)
    epsilon = epsilon_calc(H_0, phi_0, V)
    print("ε", epsilon)
    print("dφ", dphi)

    if track_N_time:
        N_arr = np.zeros_like(phi_arr)
        epsilon_arr = np.zeros_like(phi_arr)
        epsilon_arr[0] = epsilon

    # Main loop
    n=0
    while (epsilon < 1) and (n < max_n):
        # Perform RK4 step
        phi_arr[n + 1], H_arr[n + 1] = rk4_step(phi_arr[n], H_arr[n], step=dphi, derivative=HJDerivative(V))
        # Update epsilon
        epsilon = epsilon_calc(H_arr[n + 1], phi_arr[n + 1], V)
        print("ε", epsilon)

        if track_N_time:
            epsilon_arr[n + 1] = epsilon
            N_arr[n + 1] = N_arr[n] + dphi * (
                integrand_for_N(epsilon) + integrand_for_N(epsilon_arr[n])
            ) / 2
        
        # Adapt step size
        dphi = -adapt_step_factor(H_arr[n + 1], phi_arr[n + 1], dphi_0, V=V)
        print("dφ", dphi)

        n += 1

    if track_N_time:
        return phi_arr[:n + 1], H_arr[:n + 1], N_arr[:n + 1], epsilon_arr[:n + 1]
    return phi_arr[:n + 1], H_arr[:n + 1], epsilon

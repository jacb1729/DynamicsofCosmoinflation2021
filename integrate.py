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
    
class NegativeHJDerivative(DerivativeBase):
    """Negative Hamilton-Jacobi derivative for inflationary dynamics."""
    def __init__(self, V: Potential = PowerPotential()):
        self.V = V

    def __call__(self, phi: float, H: float) -> float:
        # Negative Hamilton-Jacobi equation: H' = -sqrt(1.5*H^2 - 0.5*V(φ))
        return -grad_calc(H, phi, self.V)

def grad_calc(H: float, phi: float, V: Potential = PowerPotential()):
    """Calculate the magnitude of the gradient dH/dφ (used for RK4 steps)."""
    # 1.5*H^2 - 0.5*V0*phi^m corresponds to the sign-adjusted derivative (squared) of H.
    # We take the square root of its absolute value for the magnitude.
    return np.sqrt(abs(grad_sqr(H, phi, V)))

def grad_sqr(H: float, phi: float, V: Potential = PowerPotential()):
    """Calculate the squared gradient 1.5*H^2 - 0.5*V(φ)."""
    return 1.5 * (H ** 2) - 0.5 * V(phi)

def epsilon_calc(H: float, phi: float, V: Potential = PowerPotential()):
    """Calculate the slow-roll parameter ε = 2|H'|/H (squared form used for efficiency)."""
    # ε = 2 * |(dH/dφ)| / H, where (dH/dφ)^2 = 1.5*H^2 - 0.5*V(φ) (Hamilton-Jacobi equation)
    return 2 * abs(grad_sqr(H, phi, V=V)) / (H ** 2)

def sr_epsilon_calc(phi: float, V: Potential = PowerPotential()):
    """Calculate the slow-roll parameter ε in the slow-roll approximation."""
    return 0.5 * (V.derivative(phi) / V(phi))**2

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

        if track_N_time:
            epsilon_arr[n + 1] = epsilon
            N_arr[n + 1] = N_arr[n] + abs(dphi) * (
                integrand_for_N(epsilon) + integrand_for_N(epsilon_arr[n])
            ) / 2
        
        # Adapt step size
        dphi = -adapt_step_factor(H_arr[n + 1], phi_arr[n + 1], dphi_0, V=V)

        n += 1

    if track_N_time:
        return phi_arr[:n + 1], H_arr[:n + 1], N_arr[:n + 1], epsilon_arr[:n + 1]
    return phi_arr[:n + 1], H_arr[:n + 1], epsilon

def get_trajectory_increasing_phi(V: Potential, phi_0: float, H_0: float, dphi_0 = 0.0001,
                                  bound_multiplier=100, suppress=False, sr_epsilon_limit=0.001,
                                  N_bound=60):
    """Get the trajectory of H against φ for an increasing φ field under the potential V(φ).
    If the slow roll parameter ε drops below sr_epsilon_limit, we switch to slow-roll approximation,
    then the signal of dφ is reversed to decrease φ until ε >= 1 again."""

    # We check whether the initial point is in the right domain
    if not (3 * (H_0 ** 2) > V(phi_0) > 2 * (H_0 ** 2) or suppress):
        raise ValueError("Initial conditions are not feasible")

    # Initialize variables
    max_n = bound_multiplier * int(abs(phi_0 / dphi_0))
    phi_arr = np.zeros(max_n + 1)
    H_arr = np.zeros_like(phi_arr)
    ε_arr = np.zeros_like(phi_arr)
    phi_arr[0] = phi_0
    H_arr[0] = H_0
    ε_arr[0] = epsilon_calc(H_0, phi_0, V)
    dphi = adapt_step_factor(H_0, phi_0, dphi_0, V=V)
    N_arr = np.zeros_like(phi_arr)
    n_sgn_change = np.inf  # To track where the sign change occurs

    # First loop
    # We will use the SR approximation to track when to change the regime
    # to slow roll. Below the top condition, the conditions are:
    # # ε < 1 - general end of inflation
    # # ε > sr_epsilon_limit - so to not move to SR too early
    # # sr_ε > sr_epsilon_limit - to ensure we are out of slow-roll already
    # # relative change in ε > 2% - to ensure the sr approximation is no longer valid
    n=0
    while (n < max_n)\
        and (N_arr[n] < N_bound)\
        and (ε_arr[n] < 1)\
        and ((ε_arr[n] > sr_epsilon_limit)\
        or (sr_epsilon_calc(phi_arr[n], V) > sr_epsilon_limit)\
        or abs(ε_arr[n] - sr_epsilon_calc(phi_arr[n], V)) / ε_arr[n] > 0.02):

        # Perform RK4 step
        phi_arr[n + 1], H_arr[n + 1] = rk4_step(phi_arr[n], H_arr[n], step=dphi, derivative=HJDerivative(V))
        # Update epsilon
        ε_arr[n + 1] = epsilon_calc(H_arr[n + 1], phi_arr[n + 1], V)
        # Update N
        N_arr[n + 1] = N_arr[n] + abs(dphi) * (
            integrand_for_N(ε_arr[n]) + integrand_for_N(ε_arr[n + 1])
        ) / 2

        # Adapt step size
        dphi = adapt_step_factor(H_arr[n + 1], phi_arr[n + 1], dphi_0, V=V)

        n += 1

    # Slow-roll approximation
    # H' becomes the negative of grad_calc symmetrically, fixing H
    # the sign of dφ is reversed
    if (n < max_n) and (ε_arr[n] < 1):
        H_arr[n + 1] = H_arr[n]
        phi_arr[n + 1] = phi_arr[n]
        ε_arr[n + 1] = ε_arr[n]

        print("SR Invoked at n = {}, φ = {}, H = {}, ε = {}".format(n, phi_arr[n], H_arr[n], ε_arr[n]))

        # Update N as per the quadrant crossing step
        N_arr[n + 1] = N_arr[n] + (
            (2**1.5)*grad_calc(H_arr[n], phi_arr[n], V)*((V(phi_arr[n]))**0.5)
            )/(
                (3**0.5)*(V.derivative(phi_arr[n]))
            )
        dphi = -dphi
        n += 1
        n_sgn_change = n - 1  # Mark the sign change index
    
    # Second loop - decreasing φ with sign-flipped H'
    while (n < max_n) and (ε_arr[n] < 1):
        # Perform RK4 step
        phi_arr[n + 1], H_arr[n + 1] = rk4_step(phi_arr[n], H_arr[n], step=dphi, derivative=NegativeHJDerivative(V))
        # Update epsilon
        ε_arr[n + 1] = epsilon_calc(H_arr[n + 1], phi_arr[n + 1], V)
        # Update N
        N_arr[n + 1] = N_arr[n] + abs(dphi) * (
            integrand_for_N(ε_arr[n]) + integrand_for_N(ε_arr[n + 1])
        ) / 2

        # Adapt step size
        dphi = -adapt_step_factor(H_arr[n + 1], phi_arr[n + 1], dphi_0, V=V)

        n += 1

    return phi_arr[:n + 1], H_arr[:n + 1], N_arr[:n + 1], ε_arr[:n + 1], n_sgn_change
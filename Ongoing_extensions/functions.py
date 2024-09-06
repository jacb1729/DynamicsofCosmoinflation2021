import numpy as np
from numba import njit
from collections import namedtuple

PartialTrajectory = namedtuple('PartialTrajectory', ['φ', 'H', 'n', 'epsilon'])
TrajectoryResult = namedtuple('TrajectoryResult', ['φ', 'H', 'n', 'epsilon', 'N', 'n_sgn_change'])

grad_sqr = lambda H, V0, φ, m: 1.5*(H**2) - 0.5*V0*(φ**m)
grad_calc = lambda H, V0, φ, m: (abs(grad_sqr(H, V0, φ, m)))**0.5
epsilon_calc = lambda  H, V0, φ, m: 2*(abs(grad_sqr(H, V0, φ, m))/(H**2))
g = lambda x: np.clip(x, 0.00001, None)

def f(H, V0, φ, m):
    '''Interpolation of step-size so that particularly small ε values have small enough step sizes. 
    This could be particularly helpful with ϕ^m potentials for small m>0'''
    returnval = (0.5**((0.0001/g(epsilon_calc(H, V0, φ, m))))) if epsilon_calc(H, V0, φ, m) < 0.0001 else 0.5
    return returnval

@njit
def trajectory(V_0, m, φ_0, dφ_0 = 0.0001, boundmultiplier = 100, H0 = 2, suppress = False):
    '''The first stage is to integrate forwards in time, to the end of inflation. We use and RK4 adaptive-step integrator,
    but the step size adapts according to ε rather than some error estimate.'''
    #
    #We check whether the initial point is in the right domain
    #
    if not (suppress or 3*(H0**2) > V_0*((φ_0)**m) > 2*(H0**2)):
        raise Exception("wrong parameter choice")
    #Initialise some zero arrays - we choose the maximal size dependent on the initial ϕ, initial step size (so to fit in a weak lower bound
    #number of steps before ϕ = 0) and an input 'boundmultiplier' default 100
    #
    T: int =  int(φ_0/dφ_0)
    φ: np.ndarray = np.zeros(boundmultiplier*T+1)  #adapting ϕ mesh
    H: np.ndarray = np.zeros(boundmultiplier*T+1)  #H values
    φ[0] = φ_0 #initial ϕ
    H[0]  = H0 #initial H 
    epsilon: float = epsilon_calc(H[0], V_0, φ[0], m) #initial epsilon
    #
    n: int = 0
    dφ: float = dφ_0
    #
    while (epsilon < 1) and (n < boundmultiplier*T):
        #Here we make successive gradient estimates, as per RK integration, using grad_calc to calculate the gradient at a point
        #
        K1: float = grad_calc(H[n], V_0, φ[n], m)
        K2: float = grad_calc(H[n] - 0.5*dφ*K1, V_0, φ[n] - 0.5*dφ, m)
        K3: float = grad_calc(H[n] - 0.5*dφ*K2, V_0, φ[n] - 0.5*dφ, m)
        K4: float = grad_calc(H[n] - dφ*K3, V_0, φ[n] - dφ, m)
        grad: float = (K1 + 2*K2 + 2*K3 + K4)/6
        #Currently, we are moving forwards in time, corresponding to decreasing ϕ
        #
        H[n+1] = H[n] - dφ*grad
        φ[n+1] = φ[n] - dφ
        epsilon = epsilon_calc(H[n+1], V_0, φ[n+1], m)
        coeff: float = f(H[n+1], V_0, φ[n+1], m)
        #Here, we change dφ so that its new value would not make it cross the ε = 0 contour, and our choice of coefficient hopefully ensures
        #that it doesn't cross the separatrix for this potential
        #
        dφ = min(
            coeff*(H[n+1] - ((V_0*(φ[n+1])**m)*(1/3))**0.5)/(grad_calc(H[n+1], V_0, φ[n+1], m)),
                  dφ_0
                )
        n += 1
    return PartialTrajectory(φ[:n+1], H[:n+1], n, epsilon) #after the nth value of φ, H (if such exists) the values are the initial, meaningless zero values
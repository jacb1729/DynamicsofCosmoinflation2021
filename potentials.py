import abc

class Potential(abc.ABC):
    '''Base class for inflation potentials, V:φ->V(φ)
    To make a custom potential, subclass this and implement the value method.'''
    @abc.abstractmethod
    def value(self, phi: float) -> float:
        pass

    def __call__(self, phi: float) -> float:
        return self.value(phi)

    def derivative(self, phi: float, h: float = 1e-5, last_approx: float = None) -> float:
        """Numerical derivative of the potential using central difference.
        Potential needs to be differentiable.
        Iterates until convergence to consecutive approximations being within 1e-5."""
        approx = (self.value(phi + h) - self.value(phi - h)) / (2 * h)
        if last_approx is None or abs(approx - last_approx) > 1e-5:
            return self.derivative(phi, h=h/2, last_approx=approx)
        return approx



class PowerPotential(Potential):
    '''Potentials of the form V(φ) ~ φ^m for some m'''
    def __init__(self, m = 2, V0 = 0.1):
        self.m = m
        self.V0 = V0

    def value(self, phi: float) -> float:
        """Compute the inflation potential V(φ) = V0 * φ^m."""
        return self.V0 * (phi ** self.m)
    
    # We can override derivative because we know the analytic form
    def derivative(self, phi: float) -> float:
        """Compute the derivative V'(φ) = m * V0 * φ^(m-1)."""
        return self.m * self.V0 * (phi ** (self.m - 1))
    
    def __repr__(self):
        return f"PowerPotential(V:φ -> {self.V0} * φ^{self.m})"
    
if __name__ == "__main__":
    pot = PowerPotential(m=2, V0=1)
    print(pot)
    phi = 1
    print(f"V({phi}) =", pot(phi))
    print(f"V'({phi}) =", pot.derivative(phi))
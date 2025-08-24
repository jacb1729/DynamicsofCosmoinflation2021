import abc

class Potential(abc.ABC):
    '''Base class for inflation potentials, V:φ->V(φ)'''
    @abc.abstractmethod
    def value(self, phi: float) -> float:
        pass

    def __call__(self, phi: float) -> float:
        return self.value(phi)



class PowerPotential(Potential):
    '''Potentials of the form V(φ) ~ φ^m for some m'''
    def __init__(self, m = 2, V0 = 0.1):
        self.m = m
        self.V0 = V0

    def value(self, phi: float) -> float:
        """Compute the inflation potential V(φ) = V0 * φ^m."""
        return self.V0 * (phi ** self.m)
    
    def __repr__(self):
        return f"PowerPotential(V:φ -> {self.V0} * φ^{self.m})"
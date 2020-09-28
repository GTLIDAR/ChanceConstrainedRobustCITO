import numpy as np 
from abc import ABC, abstractmethod 

class RigidBodyModel(ABC):
    # Define methods for the dynamic system properties
    @abstractmethod
    def inertia_matrix(self, q):
        pass
    @abstractmethod 
    def coriolis_matrix(self,q,dq):
        pass
    @abstractmethod
    def gravity_matrix(self,q):
        pass
    @abstractmethod
    def control_selector(self, q):
        pass

    def manipulator_dynamics(self, q, dq):
        """
        manipulator_dynamics: the inertia and force effects at a given configuration

        Syntax: M, F, B = model.manipulator_dynamics(q, dq)
        Arguments:
            q:  An Nx1 array of configuration variables
            dq: An Nx1 array of configuration rates

        Return Value: The tuple (M, F, B)
            M: An NxN matrix, the inertia matrix of the model
            F: An Nx1 array, the sum of Coriolis and Gravity forces
            B: An NxM array, the control selector matrix
        """
        # Get the inertia matrix
        M = self.inertia_matrix(q)
        # Calculate the sum of coriolis and gravity effects
        C = self.coriolis_matrix(q, dq)
        G = self.gravity_matrix(q)
        F = C.dot(dq) + G
        B = self.control_selector(q)
        return (M, F, B)

    def integrate(self, h, x, u):
        """
        Semi-implicit Euler Integration
        """
        q, dq = np.split(x, 2)
        # Integrate the velocity first
        M, F, B = self.manipulator_dynamics(q,dq)
        dq = dq + h * np.linalg.inv(M).dot(F + B.dot(u))
        # Then integrate the position
        q = q + h * dq
        # Return the new state
        return np.concatenate((q, dq))

    def simulate(self, dt, x0, u):
        """
        """
        # Initialize the simulation array
        _, N = u.shape
        nX, _ = x0.shape
        x = np.zeros(shape=(nX,N))
        x[:, 0]  = x0
        t = np.zeros(N)
        for n in range(1,N):
            x[:, n] = self.integrate(dt, x[:, n-1], u[:,n])
            t[:, n] = t[:,n -1] + dt
        return (t,x)

if __name__ == "__main__":
    print('Hello world')
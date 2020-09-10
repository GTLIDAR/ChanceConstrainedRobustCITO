import numpy as np 
from abc import ABC, abstractmethod


class ContactDynamics(ABC, MechanicalModel):
    @abstractmethod
    def contact_jacobian(self, q):
        pass 
    @abstractmethod 
    def contact_normal(self,q):
        pass
    
    # Common methods across all contact models
    def integrate(self, h, x, u):
        # Split configuration from velocity
        L, _ = x.shape
        nQ = int(L/2)
        q = x[0:nQ]
        dq = x[nQ:]
        # Calculate contact force
        f = self.contact_force(h, q, dq, u)
        # Get contact Jacobian
        J = self.contact_jacobian
        # Get dynamics
        M, C, B = self.manipulator_dynamics(q, dq)
        # Integrate velocity
        dq = dq + h*np.linalg.inv(M).dot(B.dot(u) - C + J.dot(f))
        # Integrate the position
        q = q + h * dq
        # Return the integrated state
        return np.concatenate((q, dq))

    def dynamics(self, x, f):
        """
        dynamics: update the state using a semi-implicit timestepping scheme
        """
        # Get the configuration and the velocity
        L = x.size
        q = x[0:int(L/2)]
        qdot = x[int(L/2):]
        # Estimate the next configuration, assuming constant velocity
        qhat = q + self.timestep * qdot 
        # Get the current system properties
        M = self.inertia_matrix(qhat)
        C = self.coriolis_matrix(qhat, qdot)
        N = self.gravity_matrix(qhat)
        Jt, Jn = self.contact_jacobian(qhat) 
        J = np.concatenate((Jt, Jn), axis=1)
        # Calculate the next state
        b = self.timestep * (J.dot(f) - C.dot(qdot) - N) + M.dot(qdot)
        qdot = np.linalg.solve(M,b)
        q += self.timestep * qdot
        # Collect the configuration and velocity into a state vector
        x = np.concatenate((q,qdot), axis=0)
        return x

    def contacts(self, x):
        """
        contacts: solve for the contact forces using Quadratic Programming (QP) 
        contacts models the ground contact conditions using Linear Complementarity Problems (LCPs)
        """
        # Get the configuration and generalized velocity
        L = x.size
        q = x[0:int(L/2)].transpose()
        qdot = x[int(L/2):].transpose()
        # Estimate the configuration at the next time step
        qhat = q + self.timestep*qdot
        # Get the system parameters
        M = self.inertia_matrix(qhat)
        C = self.coriolis_matrix(qhat,qdot)
        N = self.gravity_matrix(qhat)
        Jt, Jn = self.contact_jacobian(qhat) 
        normals, alpha = self.contact_normal(qhat)
        # Calculate the QP size from the contact Jacobian
        numT = Jt.shape[1]
        numN = Jn.shape[1]
        S = numT + 2*numN
        # Calculate the offset vector for the LCP problem
        z = np.zeros(shape=(S,), dtype=float)
        tau = self.timestep * (C.dot(qdot) + N) - M.dot(qdot)
        b = np.linalg.solve(M,tau)
        z[0:numN] = -self.timestep * normals.dot(b) + normals.dot(q) - alpha
        z[numN:numT+numN] = -1 * Jt.transpose().dot(b)
        # Calculate the problem matrix for the QP         
        P = np.zeros(shape=(S,S), dtype=float)
        u = np.ones(shape=(numN, numT)) # A vector of ones
        Mn = np.linalg.solve(M,Jn)
        Mt = np.linalg.solve(M,Jt)
        P[0:numN, 0:numN] = self.timestep**2 * normals.dot(Mn)
        P[0:numN, numN:numN+numT] = self.timestep**2 * normals.dot(Mt)
        P[numN:numN+numT, 0:numN] = self.timestep * Jt.transpose().dot(Mn)
        P[numN:numN + numT, numN:numN+numT] = self.timestep * Jt.transpose().dot(Mt)
        P[numN:numN + numT, numN+numT:] = u.transpose()
        P[numN + numT:, 0:numN] = np.identity(numN) * self.friction 
        P[numN + numT:, numN:numN+numT] = -1 * u
        """       
        # Solve the LCP using Lemke's Algorithm

        if sol == 1:
            print('No contact solution found')
            print(msg)
            return np.zeros(shape=(1,numN+numT), dtype=float)
        else:
            fN = w[0:numN]
            fT = w[numN:numN+numT]
            return np.concatenate((fT, fN), axis=0) 
        """

    def simulate(self, x0, T):
        """
        simulate: generates a trajectory of the system starting at x0 from t = [0,T)
        """
        # Pre-initialize arrays
        nX = x0.size 
        time = np.arange(0,T,self.timestep)
        nT = time.shape[0]
        X = np.zeros(shape=(nX,nT), dtype=float)
        X[:,0] = x0
        # Determine the size of the contact force
        Jt, Jn = self.contact_jacobian(X[0:int(nX/2),0])
        nF = Jt.shape[1] + Jn.shape[1]
        F = np.zeros(shape=(nF,nT), dtype=float)
        # Run the main simulation loop
        for n in range(1,nT):
            # Calculate the contact forces
            F[:,n] = self.contacts(X[:,n-1])
            # Calculate the updated state vector
            X[:,n] = self.dynamics(X[:,n-1], F[:,n])

        # Return a trajectory with the time, states, and contact forces
        return (time, X, F)
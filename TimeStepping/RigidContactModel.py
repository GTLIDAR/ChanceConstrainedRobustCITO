import numpy as np 
import lemkelcp as lcp 

class RigidContactModel():
    def __init__(self, plant, terrain):
        self.plant = plant
        self.terrain = terrain  

    def contact_jacobian(self, q):
        J = self.plant.jacobian(q)
        x = self.plant.kinematics(q)
        Jn = []
        Jt = []
        for n in range(0, len(x)):
            p = x[n]
            Jp = J[n]
            y = self.terrain.nearest(p)
            N, T = self.terrain.basis(y)
            Jn.append(N.transpose().dot(Jp))
            Jt.append(T.transpose().dot(Jp))
            Jt.append(-T.transpose().dot(Jp))
        return (np.vstack(Jn), np.vstack(Jt))

    def contact_distance(self, q):

        x = self.plant.kinematics(q)
        phi = []
        for p in x:
            y = self.terrain.nearest(p)
            N, _ = self.terrain.basis(y)
            phi.append(N.transpose().dot(p - y))

        return np.array(phi)

    def integrate(self, h, x, u):
        # Split configuration from velocity
        q, dq = np.split(x, 2)
        # Calculate contact force
        f = self.contact_force(h, q, dq, u)
        # Get contact Jacobian
        Jn, Jt = self.contact_jacobian(q)
        J = np.concatenate((Jn, Jt), axis = 0)
        # Get dynamics
        M, C, B = self.plant.manipulator_dynamics(q, dq)
        # Integrate velocity
        dq = dq + h*np.linalg.inv(M).dot(B.dot(u) - C + J.transpose().dot(f))
        # Integrate the position
        q = q + h * dq
        # Return the integrated state
        return np.concatenate((q, dq))

    def dynamics(self, h, x, u, f):
        """
        dynamics: update the state using a semi-implicit timestepping scheme
        """
        # Get the configuration and the velocity
        q, dq = np.split(x,2)
        # Estimate the next configuration, assuming constant velocity
        qhat = q + h * dq
        # Get the current system properties
        M = self.plant.inertia_matrix(qhat)
        C = self.plant.coriolis_matrix(qhat, dq)
        N = self.plant.gravity_matrix(qhat)
        B = self.plant.control_selector(qhat)
        Jn, Jt = self.contact_jacobian(qhat) 
        J = np.vstack((Jn, Jt))
        # Calculate the next state
        b = h * (B.dot(u) - C.dot(dq) - N) + J.transpose().dot(f)
        v = np.linalg.solve(M,b)
        dq += v
        q += h * dq
        # Collect the configuration and velocity into a state vector
        return np.concatenate((q,dq), axis=0)

    def contacts(self, h, x, u):
        """
        contacts: solve for the contact forces using LCP
        contacts models the ground contact conditions using Linear Complementarity Problems (LCPs)
        """
        # Get the configuration and generalized velocity
        q, dq = np.split(x,2)
        # Estimate the configuration at the next time step
        qhat = q + h*dq
        # Get the system parameters
        M, C, B = self.plant.manipulator_dynamics(qhat, dq)
        tau = B.dot(u) - C
        Jn, Jt = self.contact_jacobian(qhat) 
        phi = self.contact_distance(qhat)
        alpha = Jn.dot(qhat) - phi
        # Calculate the force size from the contact Jacobian
        numT = Jt.shape[0]
        numN = Jn.shape[0]
        S = numT + 2*numN
        # Initialize LCP parameters
        P = np.zeros(shape=(S,S), dtype=float)
        u = np.zeros(shape=(numN, numN*numT))
        numF = int(numT/numN)
        for n in range(0, numN):
            u[n, n*numF + numN:numN + (n+1)*numF] = 1
        # Construct LCP matrix
        J = np.vstack((Jn, Jt))
        JM = J.dot(np.linalg.inv(M))
        P[0:numN + numT, 0:numN + numT] = JM.dot(J.transpose())
        P[:, numN + numT:] = u.transpose()
        P[numN + numT:, :] = -u
        P[numN + numT:, 0:numN] = np.identity(numN) * self.terrain.friction 
        #Construct LCP bias vector
        z = np.zeros(shape=(S,), dtype=float)
        z[0:numN+numT] = h * JM.dot(tau) + J.dot(dq)
        z[0:numN] += (Jn.dot(q) - alpha)/h   
        # Solve the LCP for the reaction impluses
        sol = lcp.lemkelcp(P, z)
        f, code, _ = sol
        if f is None:
            return np.zeros(shape=(numN+numT,))
        else:
            # Strip the slack variables from the LCP solution
            return f[0:numN + numT]

    def simulate(self, x0, T, h = 0.01, u = None):
        """
        simulate: generates a trajectory of the system starting at x0 from t = [0,T)
        """
        # Pre-initialize arrays
        nX = x0.size 
        time = np.arange(0,T,h)
        nT = time.shape[0]
        x = np.zeros(shape=(nX,nT), dtype=float)
        x[:,0] = x0
        q0, dq0 = np.split(x0, 2)
        if u is None:
            B = self.plant.control_selector(q0)
            u = np.zeros(shape=(B.shape[0], nT), dtype=float)
        # Determine the size of the contact force
        Jt, Jn = self.contact_jacobian(q0)
        nF = Jt.shape[1] + Jn.shape[1]
        f = np.zeros(shape=(nF,nT), dtype=float)
        # Run the main simulation loop
        for n in range(1,nT):
            # Calculate the contact forces
            f[:,n] = self.contacts(h, x[:,n-1], u[:,n])
            # Calculate the updated state vector
            x[:,n] = self.dynamics(h, x[:,n-1], u[:,n], f[:,n])
        # Return a trajectory with the time, states, and contact forces
        return (time, x, f)

if __name__ == "__main__":
    print('Hello world')
import numpy as np
from numpy import sin, cos
from TimeStepping.contactDynamics import ContactDynamics
import matplotlib.pyplot as plt 
from math import pi

class Rod(ContactDynamics):
    def __init__(self, m=1, l=1, r=1, J=1, h=1,mu=1):
        self.mass = m
        self.length = l
        self.radius = r
        self.inertia = J
        self.timestep = h
        self.friction = mu

    def inertia_matrix(self, _):
        M = np.zeros(shape=(3,3), dtype=float)
        M[0,0] = self.mass
        M[1,1] = self.mass
        M[2,2] = self.inertia
        return M

    def coriolis_matrix(self,q,qdot):
        return np.zeros(shape=(3,3), dtype=float)        

    def gravity_matrix(self, _):
        N = np.zeros(shape=(3,), dtype=float)
        N[1] = self.mass * 9.81
        return N
    
    def contact_jacobian(self,q):
        # Calculate the tangential component of the contact Jacobian, Jt
        Jt = np.zeros(shape=(3,4), dtype=float)
        Jn = np.zeros(shape=(3,2), dtype=float)
        Jt[0,:] = [1, -1, 1, -1]
        Jt[2,:] = [1, -1, -1, 1] 
        Jt[2,:] = self.length/2 * sin(q[2]) * Jt[2,:]
        # Calculate the normal component, Jn
        Jn[1,:] = [1,1]
        Jn[2,:] = [-cos(q[2]), cos(q[2])]
        Jn[2,:] = self.length/2 * Jn[2,:]
        # Return the two Jacobians in a tuple
        return (Jt, Jn)

    def contact_normal(self, q):
        """
        returns the contact normal n and the location parameter alpha 
        for the contact nonpenetration constraint evaluated at 
        configuration q
        """
        # Calculate the contact normal
        n = np.zeros(shape=(2,3), dtype=float)
        n[0,:] = [0, 1, -self.length/2*cos(q[2])]
        n[1,:] = [0, 1, self.length/2*cos(q[2])]
        # Calculate the location parameter
        alpha = np.zeros(shape=(2,), dtype=float)
        alpha[0] = self.radius + self.length/2 * (sin(q[2]) - q[2]*cos(q[2]))
        alpha[1] = self.radius + self.length/2 * (q[2]*cos(q[2]) - sin(q[2]))
        return (n, alpha)

    def plotTrajectory(self, t, x, f):
        plt.figure(1)
        labels = ['Horizontal','Vertical','Angular']
        axes = [0,0,0]
        # Make plots of the angular velocity
        for n in range(1,4):
            axes[n-1] = plt.subplot(3,1,n)
            plt.plot(t, x[n+2,:])
            plt.ylabel(labels[n-1])
        plt.xlabel('Time (s)')
        # add a title to the first axis
        plt.sca(axes[0])
        plt.title('Velocities')

        # Resolve the contact forces
        cF = np.zeros(shape=(4,f.shape[1]),dtype=float)
        cF[0,:] = f[0,:] - f[1,:]
        cF[1,:] = f[2,:] - f[3,:]
        cF[2,:] = f[4,:]
        # Make a plot of the contact forces
        plt.figure(2)
        fLabels = ['Friction 1','Friction 2','Normal 1','Normal 2']
        fAxes = [0,0,0,0]
        for n in range(1,5):
            fAxes[n-1] = plt.subplot(4,1,n)
            plt.plot(t,cF[n-1,:])
            plt.ylabel(fLabels[n-1])
        # Set the xlabel
        plt.xlabel('Time (s)')
        # Add a plot title
        plt.sca(fAxes[0])
        plt.title('Contact Forces')

        plt.show()

        # Show the figures

    def draw(self, ax, q):
        x1, y1, x2, y2 = self.cartesian(q)
        plt.sca(ax)
        plt.plot([x1,x2], [y1,y2])

    def cartesian(self, q):
        """
        cartesian: returns the endpoints of the rod in cartesian coordinates
        """
        x1 = q[0] + self.length/2 * cos(q[2]) + self.radius
        x2 = q[0] - self.length/2 * cos(q[2]) - self.radius
        y1 = q[1] + self.length/2 * sin(q[2]) + self.radius
        y2 = q[1] - self.length/2 * sin(q[2]) - self.radius
        return (x1,y1,x2,y2)

if __name__ == '__main__':
    model = Rod(m=1, l=0.5, r=0.05, J=0.002, h=0.0025, mu=0.6)
    # Check the values of the model parameters
    print(model.inertia_matrix([0,0,0]))
    print(model.gravity_matrix([0,0,0]))
    print(model.contact_jacobian([0,0,0]))
    # Run a test simulation
    X0 = np.array([0.0, 3.0, pi/6, 0.0, 0.0, 4.0])
    t,x,f = model.simulate(X0, 1.5)
    # Plot the trajectory
    model.plotTrajectory(t,x,f)

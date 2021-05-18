import numpy as np
import matplotlib.pyplot as plt 
from numpy import sin, cos
from math import pi

from TimeStepping.RigidBodyModel import RigidBodyModel

class Rod(RigidBodyModel):
    def __init__(self, m=1, l=0.5, r=0.05, J=0.02):
        self.mass = m
        self.length = l
        self.radius = r
        self.inertia = J

    def inertia_matrix(self, q):
        return np.diag([self.mass, self.mass, self.inertia])

    def coriolis_matrix(self,q, dq):
        return np.zeros(shape=(3,3), dtype=float)        

    def gravity_matrix(self, q):
        return np.array([0, self.mass * 9.81, 0])
    
    def control_selector(self, q):
        return np.array([0, 0, 0])

    def kinematics(self, q):
        p1 = np.array([q[0] + self.length/2*cos(q[2]) + self.radius, q[1] + self.length/2*sin(q[2]) - self.radius])
        p2 = np.array([q[0] - self.length/2*cos(q[2]) - self.radius, q[1] - self.length/2*sin(q[2]) - self.radius])
        return [p1, p2]

    def jacobian(self,q):
        # Calculate the Jacobians for the endpoints of the rod
        J1 = np.array([[1, 0, -self.length/2*sin(q[2])],[0, 1, self.length/2*cos(q[2])]])
        J2 = np.array([[1, 0, self.length/2*sin(q[2])],[0, 1, -self.length/2*cos(q[2])]])
        return [J1, J2]

    def plotTrajectory(self, t, x, f):
        plt.figure(1)
        labels = ['Horizontal','Vertical','Angular']
        axes = [0,0,0]
        # Make plots of the positions
        for n in range(0,3):
            axes[n] = plt.subplot(3,1,n+1)
            plt.plot(t, x[n,:])
            plt.ylabel(labels[n])
        plt.xlabel('Time (s)')
        plt.sca(axes[0])
        plt.title('Positions')
        # Plot Velocities
        plt.figure(2)
        vaxes = [0,0,0]
        for n in range(1,4):
            vaxes[n-1] = plt.subplot(3,1,n)
            plt.plot(t, x[n+2,:])
            plt.ylabel(labels[n-1])
        plt.xlabel('Time (s)')
        # add a title to the first axis
        plt.sca(vaxes[0])
        plt.title('Velocities')

        # Resolve the contact forces
        cF = np.zeros(shape=(4,f.shape[1]),dtype=float)
        cF[0,:] = f[0,:] 
        cF[1,:] = f[1,:]
        cF[2,:] = f[2,:] - f[3,:]
        cF[3,:] = f[4,:] - f[5,:]
        # Make a plot of the contact forces
        plt.figure(3)
        fLabels = ['Normal 1','Normal 2','Friction 1','Friction 2']
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
    print("Hello world")
    # model = Rod(m=1, l=0.5, r=0.05, J=0.002, h=0.0025, mu=0.6)
    # # Check the values of the model parameters
    # print(model.inertia_matrix([0,0,0]))
    # print(model.gravity_matrix([0,0,0]))
    # print(model.contact_jacobian([0,0,0]))
    # # Run a test simulation
    # X0 = np.array([0.0, 3.0, pi/6, 0.0, 0.0, 4.0])
    # t,x,f = model.simulate(X0, 1.5)
    # # Plot the trajectory
    # model.plotTrajectory(t,x,f)

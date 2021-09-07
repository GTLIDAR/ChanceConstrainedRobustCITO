"""
Classes and methods for creating and visualizing the sliding block

Luke Drnach
January 14, 2021
"""
import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import RigidTransform, PiecewisePolynomial
# Project-specific imports
from utilities import FindResource, GetKnotsFromTrajectory, quat2rpy
from systems.timestepping import TimeSteppingMultibodyPlant
from systems.visualization import Visualizer
from systems.terrain import FlatTerrain
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

class Block(TimeSteppingMultibodyPlant):
    def __init__(self, urdf_file="systems/block/urdf/sliding_block.urdf", terrain=FlatTerrain()):

        # Initialize the time-stepping multibody plant
        super(Block, self).__init__(file=FindResource(urdf_file), terrain=terrain)
        # Weld the center body frame to the world frame
        body_inds = self.multibody.GetBodyIndices(self.model_index[0])
        base_frame = self.multibody.get_body(body_inds[0]).body_frame()
        self.multibody.WeldFrames(self.multibody.world_frame(), base_frame, RigidTransform())

    def visualize_pyplot(self, trajectory=None):
        if type(trajectory) is np.ndarray:
            t = np.linspace(start=0, stop=1, num=trajectory.shape[1])
            trajectory = PiecewisePolynomial.FirstOrderHold(t, trajectory)
        if type(trajectory) is not PiecewisePolynomial:
            raise ValueError("trajectory must be a PiecewisePolynomial or 2D numpy array")
        return BlockPyPlotAnimator(self, trajectory)
        
    def plot_trajectories(self, xtraj=None, utraj=None, ftraj=None):
        """
        plot the state, control, and force trajectories for the Block
        
        Arguments:
            xtraj, utraj, and ftraj should be pyDrake PiecewisePolynomials
        """
        #TODO: generalize for multiple contact points, generalize for free rotation
        # Plot the State trajectories
        if xtraj is not None:
            self.plot_state_trajectory(xtraj, show=False)
        # Plot the controls
        if utraj is not None:
            self.plot_control_trajectory(utraj, show=False)
        # Plot the reaction forces
        if ftraj is not None:
            self.plot_force_trajectory(ftraj, show=False)
        # Show the plots only when one of the inputs is not None
        if xtraj is not None or utraj is not None or ftraj is not None:
            plt.show()

    @staticmethod
    def plot_state_trajectory(xtraj, show=True):
        t, x = GetKnotsFromTrajectory(xtraj)
        fig, axs = plt.subplots(2,1)
        axs[0].plot(t, x[0,:], linewidth=1.5, label='horizontal')
        axs[0].plot(t, x[1,:], linewidth=1.5, label='vertical')
        axs[0].set_ylabel('Position (m)')
        axs[1].plot(t, x[2,:], linewidth=1.5, label='horizontal')
        axs[1].plot(t, x[3,:], linewidth=1.5, label='vertical')
        axs[1].set_ylabel('Velocity (m/s)')
        axs[1].set_xlabel('Time (s)')
        axs[0].legend()
        if show:
            plt.show()
        return (fig, axs)

    @staticmethod
    def plot_control_trajectory(utraj, show=True):
        t, u = GetKnotsFromTrajectory(utraj)
        fig, axs = plt.subplots(2,1)
        axs[0].plot(t, u[0,:], linewidth=1.5)
        axs[0].set_ylabel('Control (N)')
        axs[0].set_xlabel('Time (s)')
        if show:
            plt.show()
        return (fig, axs)

    @staticmethod
    def plot_force_trajectory(ftraj, show=True):
        t, f = GetKnotsFromTrajectory(ftraj)
        fig, axs = plt.subplots(3,1)
        axs[0].plot(t, f[0,:], linewidth=1.5)
        axs[0].set_ylabel('Normal')
        axs[0].set_title('Ground reaction forces')
        axs[1].plot(t, f[1,:] - f[3,:], linewidth=1.5)
        axs[1].set_ylabel('Friction-x')
        axs[2].plot(t, f[2, :] - f[4,:], linewidth=1.5)
        axs[2].set_ylabel('Friction-y')
        axs[2].set_xlabel('Time (s)')
        if show:
            plt.show()
        return (fig, axs)

    @staticmethod
    def visualize(trajectory=None):
        vis = Visualizer("systems/block/urdf/sliding_block.urdf")
        #Weld the center body frame to the world frame
        body_inds = vis.plant.GetBodyIndices(vis.model_index)
        base_frame = vis.plant.get_body(body_inds[0]).body_frame()
        vis.plant.WeldFrames(vis.plant.world_frame(), base_frame, RigidTransform())
        # Make the visualization
        vis.visualize_trajectory(trajectory)

#TODO: Implement a multibody pass through for the free floating block
class FreeFloatingBlock(TimeSteppingMultibodyPlant):
    def __init__(self, urdf_file="systems/block/urdf/free_block.urdf", terrain=FlatTerrain()):
        # Initialize the time-stepping multibody plant
        super(FreeFloatingBlock, self).__init__(file=FindResource(urdf_file), terrain=terrain)

    @staticmethod
    def visualize(trajectory):
        # Create a visualizer
        vis = Visualizer("systems/block/urdf/free_block.urdf")
        # Make the visualization
        vis.visualize_trajectory(trajectory)

    @staticmethod
    def plot_control_trajectory(utraj, show=True):
        t, u = GetKnotsFromTrajectory(utraj)
        fig, axs = plt.subplots(2,1)
        axs[0].plot(t, u[0,:], linewidth=1.5)
        axs[0].set_ylabel('Control Force-X (N)')
        axs[1].set_ylabel('Control Force-Y (N)')
        axs[1].set_xlabel('Time (s)')
        if show:
            plt.show()
        return (fig, axs)

    def plot_state_trajectory(self, xtraj):
        t, x = GetKnotsFromTrajectory(xtraj)
        nq = self.multibody.num_positions()
        q, v = np.split(x, [nq])
        # Get orientation from quaternion
        q[1:4] = quat2rpy(q[0:4,:])
        # Plot Orientation and Position
        _, paxs = plt.subplots(2,1)
        labels=[["Roll", "Pitch", "Yaw"],["X", "Y", "Z"]]
        ylabels = ["Orientation","Position"]
        for n in range(2):
            for k in range(3):
                paxs[n].plot(t, q[1 + 3*n + k,:], linewidth=1.5, label=labels[n][k])
            paxs[n].set_ylabel(ylabels[n])
            paxs[n].legend()
        paxs[-1].set_xlabel('Time (s)')
        paxs[0].set_title("COM Configuration")
        #Plot COM orientation rate and translational velocity
        _, axs = plt.subplots(2,1)
        for n in range(2):
            for k in range(3):
                axs[n].plot(t, v[3*n + k,:], linewidth=1.5, label=labels[n][k])
            axs[n].set_ylabel(ylabels[n] + " Rate")
            axs[n].legend()
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title("COM Velocities")
    
    @staticmethod
    def plot_force_trajectory(ftraj, show=True):
        return Block.plot_force_trajectory(ftraj, show)

    def plot_trajectories(self, xtraj=None, utraj=None, ftraj=None):
        """
        plot the state, control, and force trajectories for the Block
        
        Arguments:
            xtraj, utraj, and ftraj should be pyDrake PiecewisePolynomials
        """
        #TODO: generalize for multiple contact points, generalize for free rotation
        # Plot the State trajectories
        if xtraj is not None:
            self.plot_state_trajectory(xtraj, show=False)
        # Plot the controls
        if utraj is not None:
            self.plot_control_trajectory(utraj, show=False)
        # Plot the reaction forces
        if ftraj is not None:
            self.plot_force_trajectory(ftraj, show=False)
        # Show the plots only when one of the inputs is not None
        if xtraj is not None or utraj is not None or ftraj is not None:
            plt.show()

class BlockPyPlotAnimator(animation.TimedAnimation):
    #TODO: Calculate viewing limits from trajectory
    #TODO: Get height and width of the block from the plant
    def __init__(self, plant, xtraj):
        # Store the plant, data, and key
        self.plant = plant
        _, x = GetKnotsFromTrajectory(xtraj)
        self.xtraj = x
        # Calculate the terrain height along the trajectory
        height = np.zeros((x.shape[1],))
        for n in range(0,x.shape[1]):
            pt = plant.terrain.nearest_point(x[0:3,n])
            height[n] = pt[2]
        # Create the figure
        self.fig, self.axs = plt.subplots(2,1)
        self.axs[0].set_ylabel('Terrain Height')
        self.axs[1].set_ylabel('Friction')
        self.axs[1].set_xlabel('Position')
        # Initialize the block
        self.block = Rectangle(xy=(x[0,0], x[1,0]), width=1.0, height=1.0)
        # Draw the true terrains
        self.height_true = Line2D(x[0,:], height, color='black', linestyle='-',linewidth=2.0)
        # Add all the lines to their axes
        self.axs[0].add_patch(self.block)
        self.axs[0].add_line(self.height_true)
        # Set the axis limits
        self.axs[0].set_xlim(-0.5,5.5)
        self.axs[0].set_ylim(-1.0,2.0)
        # Setup the initial animation
        animation.TimedAnimation.__init__(self, self.fig, interval=50, repeat=False, blit=True)
    
    def _draw_frame(self, framedata):
        i = framedata
        # update the block position
        xpos = self.xtraj[0,i] - 0.5
        ypos = self.xtraj[1,i] - 0.5
        self.block.set_xy((xpos, ypos))
        # Update the drawn artists
        self._drawn_artists = [self.block]

    def new_frame_seq(self):
        # Fix this
        return iter(range(self.xtraj.shape[1]))

    def _init_draw(self):
        pass

if __name__ == "__main__":
    block = Block()
    block.visualize()        
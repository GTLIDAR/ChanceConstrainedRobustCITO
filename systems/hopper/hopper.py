"""
Classes and methods for creating and visualizing a footed hopper

Luke Drnach
April 16, 2021
"""

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import RigidTransform, PiecewisePolynomial
# Project-specific imports
from utilities import FindResource, GetKnotsFromTrajectory, quat2rpy, load
from systems.timestepping import TimeSteppingMultibodyPlant
from systems.visualization import Visualizer
from systems.terrain import FlatTerrain
from pydrake.all import PiecewisePolynomial, InverseKinematics, Solve
import decorators as deco
import utilities as utils

class Hopper(TimeSteppingMultibodyPlant):
    def __init__(self, urdf_file="systems/hopper/urdf/footedhopper.urdf", terrain=FlatTerrain()):
        # Initialize
        super(Hopper, self).__init__(file=FindResource(urdf_file), terrain=terrain)
        # Weld base to world frame
        body_inds = self.multibody.GetBodyIndices(self.model_index[0])
        base_frame = self.multibody.get_body(body_inds[0]).body_frame()
        self.multibody.WeldFrames(self.multibody.world_frame(), base_frame, RigidTransform())

    def _standing_pose(self, base_pose):
        # Heel position
        xh = base_pose[0]
        yh = 0.
        # Relative heel position
        r = np.array([xh - base_pose[0], yh - base_pose[1]])
        # Knee angle
        q = np.zeros((5,))
        q[3] = -np.arccos(1 - (r[0]**2 + r[1]**2)/2)
        #Hip angle
        beta = np.arctan2(r[1], r[0])
        alpha = np.arctan2(np.sin(q[3]), 1 + np.cos(q[3]))
        q[2] = np.pi - (beta + alpha)
        # Ankle angle
        q[4] = - q[2] - q[3] + np.pi/2
        # Wrap to Pi
        for n in range(2,5):
            q[n] = (q[n] + np.pi) % (2*np.pi) - np.pi
        q[:2] = base_pose[:]
        return q

    def standing_pose_ik(self, base_pose, guess=None):
        # Create the IK problem
        IK = InverseKinematics(self.multibody, with_joint_limits=True)
        # Set default pose
        context = self.multibody.CreateDefaultContext()
        q_0 = np.zeros((self.multibody.num_positions(),))
        q_0[:2] = base_pose
        self.multibody.SetPositions(context, q_0)
        # Constrain foot positions
        world = self.multibody.world_frame()
        for pose, frame, radius in zip(self.collision_poses, self.collision_frames, self.collision_radius):
            point = pose.translation().copy()
            point[0] -= radius
            point_w = point.copy()
            point_w[0] = -point_w[-1]
            point_w[-1] = 0.
            # point_w = self.multibody.CalcPointsPositions(context, frame, point, world)
            # point_w[-1] = 0.
            IK.AddPositionConstraint(frame, point, world, point_w, point_w)
        # Set base position as a constraint
        q_vars = IK.q()
        prog = IK.prog()
        prog.AddLinearEqualityConstraint(Aeq = np.eye(2), beq = np.expand_dims(base_pose, axis=1), vars = q_vars[:2])
        # Solve the problem
        if guess is None:
            guess = self._standing_pose(base_pose)

        prog.SetInitialGuess(q_vars, guess)

        result = Solve(prog)
        # Return the configuration vector
        return result.GetSolution(IK.q()), result.is_success()

    @staticmethod
    def visualize(trajectory):
        vis = Visualizer("systems/hopper/urdf/footedhopper.urdf")
        body_inds = vis.plant.GetBodyIndices(vis.model_index)
        base_frame = vis.plant.get_body(body_inds[0]).body_frame()
        vis.plant.WeldFrames(vis.plant.world_frame(), base_frame, RigidTransform())
        vis.visualize_trajectory(trajectory)

    def plot_trajectories(self, xtraj, utraj, ftraj, jltraj, show=True, savename=None):
        fig1, axs1 = self.plot_base_trajectory(xtraj, show=False, savename=utils.append_filename(savename, 'Base'))
        fig2, axs2 = self.plot_joint_trajectory(xtraj, utraj, jltraj, show=False, savename=utils.append_filename(savename,'JointAngles'))
        fig3, axs3 = self.plot_force_trajectory(ftraj, show=False, savename=utils.append_filename(savename, 'ReactionForces'))
        if show:
            plt.show()
        return np.array([fig1, fig2, fig3]), np.concatenate([axs1, axs2, axs3])


    @deco.showable_fig
    @deco.saveable_fig
    def plot_base_trajectory(self, traj):
        t, x = GetKnotsFromTrajectory(traj)
        fig, axs = plt.subplots(2,1)
        directions = ['Horizontal','Vertical']
        for n in range(2):
            axs[0].plot(t, x[0+n,:], linewidth=1.5, label=directions[n])
            axs[1].plot(t, x[5+n, :], linewidth=1.5, label=directions[n])
        axs[0].set_ylabel('Position (m)')
        axs[1].set_ylabel('Velocity (m/s)')
        axs[1].set_xlabel('Time (s)')
        axs[0].legend()
        return fig, axs

    @deco.saveable_fig
    @deco.showable_fig
    def plot_joint_trajectory(self, xtraj, utraj, jltraj, show=True):
        t,x = GetKnotsFromTrajectory(xtraj)
        t,u = GetKnotsFromTrajectory(utraj)
        t,jl = GetKnotsFromTrajectory(jltraj)
        jl = self.resolve_limit_forces(jl)
        joints = ['Hip', 'Knee', 'Ankle']
        fig, axs = plt.subplots(4,1)
        for n in range(3):
            axs[0].plot(t, x[2+n,:], linewidth=1.5, label=joints[n])
            axs[1].plot(t, x[7+n,:], linewidth=1.5, label=joints[n])
            axs[2].plot(t, u[n,:], linewidth=1.5, label=joints[n])
            axs[3].plot(t, jl[n,:], linewidth=1.5, label=joints[n])
        axs[0].set_ylabel('Angle')
        axs[1].set_ylabel('Rate')
        axs[2].set_ylabel('Control Torque')
        axs[3].set_ylabel('Limit Torque')
        axs[3].set_xlabel('Time (s)')
        axs[0].legend()
        return fig, axs

    @deco.saveable_fig
    @deco.showable_fig
    def plot_force_trajectory(self, ftraj):
        t, f = GetKnotsFromTrajectory(ftraj)
        f = self.resolve_forces(f)
        fN, fT = (f[0:2,:], f[2:,:])
        fig, axs = plt.subplots(3,1)
        contacts = ['Toe', 'Heel']
        labels = ['Normal', 'Friction-X', 'Friction-Y']
        for n in range(2):
            axs[0].plot(t, fN[n,:], linewidth=1.5, label=contacts[n])
            axs[1].plot(t, fT[2*n,:], linewidth=1.5, label=contacts[n])
            axs[2].plot(t, fT[2*n+1,:], linewidth=1.5, label=contacts[n])
        for k in range(3):
            axs[k].set_ylabel(labels[k])
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title('Reaction Forces')
        axs[0].legend()
        return fig, axs

class AccessHopper(Hopper):
    def __init__(self, urdf_file="systems/hopper/urdf/single_legged_hopper.urdf", terrain=FlatTerrain()):
        super(AccessHopper, self).__init__(urdf_file, terrain)

    @staticmethod
    def visualize(trajectory):
        vis = Visualizer("systems/hopper/urdf/single_legged_hopper.urdf")
        body_inds = vis.plant.GetBodyIndices(vis.model_index)
        base_frame = vis.plant.get_body(body_inds[0]).body_frame()
        vis.plant.WeldFrames(vis.plant.world_frame(), base_frame, RigidTransform())
        vis.visualize_trajectory(trajectory)

if __name__ == '__main__':
    hopper = Hopper()
    hopper.Finalize()
    D = hopper.friction_discretization_matrix()
    print(f"Friction discretization: \n{D}")
    e = hopper.duplicator_matrix()
    print(f"Friction duplicator matrix: \n{e}")
    #data = load('data/FootedHopper/FootedHopperData.pkl')
    #traj = PiecewisePolynomial.FirstOrderHold(data['time'], data['state'])
    #Hopper.visualize(traj)
import matplotlib.pyplot as plt
import numpy as np
from systems.visualization import Visualizer
from pydrake.all import PiecewisePolynomial
from pydrake.math import RigidTransform, RollPitchYaw

def plot(x, u, l, t):
    fig1, axs1 = plt.subplots(6,1)
    # plot configurations
    axs1[0].set_title('Configuration Trajectory')
    axs1[0].plot(t, x[0,:], linewidth=1.5)
    axs1[0].set_ylabel('Base_x')
    axs1[1].plot(t, x[1,:], linewidth=1.5)
    axs1[1].set_ylabel('Base_height')
    axs1[2].plot(t, x[2,:], linewidth=1.5)
    axs1[2].set_ylabel('Angle_1')
    axs1[3].plot(t, x[3,:], linewidth=1.5)
    axs1[3].set_ylabel('Angle_2')
    axs1[4].plot(t, x[4,:], linewidth=1.5)
    axs1[4].set_ylabel('Angle_3')
    axs1[2].set_xlabel('Time (s)')
    foot_height = x[1,:] - np.cos(x[2,:]) - np.cos(x[3,:]/2)
    axs1[5].plot(t, foot_height, linewidth = 1.5)
    axs1[5].set_ylabel('foot height')
    # plot controls
    fig2, axs2 = plt.subplots(3,1)
    axs2[0].set_title('Control Trajectory')
    axs2[0].plot(t, u[0,:], linewidth=1.5)
    axs2[0].set_ylabel('Joint 1')
    axs2[1].plot(t, u[1,:], linewidth=1.5)
    axs2[1].set_ylabel('Joint 2')
    axs2[2].plot(t, u[2,:], linewidth=1.5)
    axs2[2].set_ylabel('Joint 3')
    axs2[2].set_xlabel('Time (s)')

    # # plot forces
    fig3, axs3 = plt.subplots(3,1)
    axs3[0].set_title('Ground reaction forces point 1')
    axs3[0].plot(t, l[0,:], linewidth=1.5)
    axs3[0].set_ylabel('Normal')
    axs3[1].plot(t, l[2,:] - l[4,:], linewidth=1.5)
    axs3[1].set_ylabel('Friction-x')
    axs3[2].plot(t, l[3, :] - l[5,:], linewidth=1.5)
    axs3[2].set_ylabel('Friction-y')
    # axs3[2].set_ylim(-0.5, 3)
    axs3[2].set_xlabel('Time (s)')

    fig4, axs4 = plt.subplots(3,1)
    axs4[0].set_title('Ground reaction forces point 2')
    axs4[0].plot(t, l[1,:], linewidth=1.5)
    axs4[0].set_ylabel('Normal')
    axs4[1].plot(t, l[6,:] - l[8,:], linewidth=1.5)
    axs4[1].set_ylabel('Friction-x')
    axs4[2].plot(t, l[7, :] - l[9,:], linewidth=1.5)
    axs4[2].set_ylabel('Friction-y')
    # axs4[2].set_ylim(-0.5, 3)
    axs4[2].set_xlabel('Time (s)')
    fig5, axs5 = plt.subplots(3,1)
    axs5[0].plot(t, l[0,:] + l[1,:], linewidth=1.5)
    axs5[0].set_ylabel('Normal')
    axs5[1].plot(t, l[2,:] - l[4,:] + l[6,:] - l[8,:], linewidth=1.5)
    axs5[1].set_ylabel('Friction-x')
    axs5[2].plot(t, l[3,:] - l[5,:] + l[7,:] - l[9,:], linewidth=1.5)
    axs5[2].set_ylabel('Friction-y')
    axs5[0].set_title('Total Ground force')
    plt.show()
    
def plot_CC():
    sigmas = np.array([0.3, 0.5, 0.7])
    # Plot the horizontal trajectory
    fig1, axs1 = plt.subplots(3,1)
    # nominal trajectory
    x_ref = np.loadtxt('data/single_legged_hopper/nominal_2/x.txt')
    u_ref = np.loadtxt('data/single_legged_hopper/nominal_2/u.txt')
    l_ref = np.loadtxt('data/single_legged_hopper/nominal_2/l.txt')
    t = np.loadtxt('data/single_legged_hopper/nominal_2/t.txt')
    
    ref_foot_height = x_ref[1,:] - np.cos(x_ref[2,:]) - np.cos(x_ref[3,:]/2)
    ref_normal_force = l_ref[0,:] + l_ref[1,:]
    axs1[0].set_ylim([1,3.2])
    axs1[1].set_ylim([-0.5,1.3])
    # sigma = 0.1
    x_1 = np.loadtxt('data/single_legged_hopper/erm_cc_2/x_{f}1.txt'.format(f = sigmas[0]))
    l_1 = np.loadtxt('data/single_legged_hopper/erm_cc_2/l_{f}1.txt'.format(f = sigmas[0]))
    foot_height_1 = x_1[1,:] - np.cos(x_1[2,:]) - np.cos(x_1[3,:]/2)
    normal_force_1 = l_1[0,:] + l_1[1,:]
    axs1[0].plot(t, x_1[1,:], linewidth= 2.5, label = '$\sigma = {f}$'.format(f = sigmas[0]))
    axs1[1].plot(t, foot_height_1,linewidth = 2.5)
    axs1[2].plot(t, normal_force_1, linewidth = 1.5)
    # sigma = 0.3
    x_3 = np.loadtxt('data/single_legged_hopper/erm_cc_2/x_{f}1.txt'.format(f = sigmas[1]))
    l_3 = np.loadtxt('data/single_legged_hopper/erm_cc_2/l_{f}1.txt'.format(f = sigmas[1]))
    foot_height_3 = x_3[1,:] - np.cos(x_3[2,:]) - np.cos(x_3[3,:]/2)
    normal_force_3 = l_3[0,:] + l_3[1,:]
    axs1[0].plot(t, x_3[1,:], linewidth= 2.5, label = '$\sigma = {f}$'.format(f = sigmas[1]))
    axs1[1].plot(t, foot_height_3, linewidth = 2.5)
    axs1[2].plot(t, normal_force_3, linewidth = 1.5)
    # sigma = 0.4
    x_5 = np.loadtxt('data/single_legged_hopper/erm_cc_2/x_{f}1.txt'.format(f = sigmas[2]))
    l_5 = np.loadtxt('data/single_legged_hopper/erm_cc_2/l_{f}1.txt'.format(f = sigmas[2]))
    foot_height_5 = x_5[1,:] - np.cos(x_5[2,:]) - np.cos(x_5[3,:]/2)
    normal_force_5 = l_5[0,:] + l_5[1,:]
    # axs1[0].plot(t, x_5[1,:], linewidth= 2.5, label = '$\sigma = {f}$'.format(f = sigmas[2]))
    # axs1[1].plot(t, foot_height_5,linewidth = 2.5)
    # axs1[2].plot(t, normal_force_5, linewidth = 2.5)

    axs1[0].plot(t, x_ref[1,:], 'k-', linewidth= 2.5, label = 'Reference')
    axs1[1].plot(t, ref_foot_height, 'k-', linewidth = 2.5)
    axs1[2].plot(t, ref_normal_force, 'k-', linewidth = 1.5)
    # ref_x.legend()
    # axs1[0].set_title('Chance Constraint Variation w/ ERM w/ warmstart')
    axs1[0].set_title('ERM + CC')
    axs1[0].set_ylabel('Base Height')
    # axs1[0].legend()
    axs1[1].set_ylabel('Foot Height')
    axs1[2].set_ylabel('Normal Impulse')
    # axs1[1].set_yticks([500, 250, 0, -250, -500])
    plt.show()

def plot_erm():
    fig1, axs1 = plt.subplots(3,1)
    # nominal trajectory
    x_ref = np.loadtxt('data/single_legged_hopper/nominal_2/x.txt')
    u_ref = np.loadtxt('data/single_legged_hopper/nominal_2/u.txt')
    l_ref = np.loadtxt('data/single_legged_hopper/nominal_2/l.txt')
    t = np.loadtxt('data/single_legged_hopper/nominal_2/t.txt')
    # plot(x_ref, u_ref, l_ref, t)
    ref_foot_height = x_ref[1,:] - np.cos(x_ref[2,:]) - np.cos(x_ref[3,:]/2)
    ref_normal_force = l_ref[0,:] + l_ref[1,:]
    # axs1[0].set_ylim([1,3.2])
    # axs1[1].set_ylim([-0.5,1.3])
    sigmas = np.array([0.3, 0.5, 0.7])
    # sigma = 0.1
    x_1 = np.loadtxt('data/single_legged_hopper/erm_new_2/x_{f}.txt'.format(f = sigmas[0]))
    l_1 = np.loadtxt('data/single_legged_hopper/erm_new_2/l_{f}.txt'.format(f = sigmas[0]))
    foot_height_1 = x_1[1,:] - np.cos(x_1[2,:]) - np.cos(x_1[3,:]/2)
    normal_force_1 = l_1[0,:] + l_1[1,:]
    axs1[0].plot(t, x_1[1,:], linewidth= 2.5, label = '$\sigma = {f}$'.format(f = sigmas[0]))
    axs1[1].plot(t, foot_height_1,linewidth = 2.5)
    axs1[2].plot(t, normal_force_1, linewidth = 1.5)
    # sigma = 0.3
    x_3 = np.loadtxt('data/single_legged_hopper/erm_new_2/x_{f}.txt'.format(f = sigmas[1]))
    l_3 = np.loadtxt('data/single_legged_hopper/erm_new_2/l_{f}.txt'.format(f = sigmas[1]))
    foot_height_3 = x_3[1,:] - np.cos(x_3[2,:]) - np.cos(x_3[3,:]/2)
    normal_force_3 = l_3[0,:] + l_3[1,:]
    axs1[0].plot(t, x_3[1,:], linewidth= 2.5, label = '$\sigma = {f}$'.format(f = sigmas[1]))
    axs1[1].plot(t, foot_height_3, linewidth = 2.5)
    axs1[2].plot(t, normal_force_3, linewidth = 1.5)
    # sigma = 0.4
    x_5 = np.loadtxt('data/single_legged_hopper/erm_new_2/x_{f}.txt'.format(f = sigmas[2]))
    l_5 = np.loadtxt('data/single_legged_hopper/erm_new_2/l_{f}.txt'.format(f = sigmas[2]))
    foot_height_5 = x_5[1,:] - np.cos(x_5[2,:]) - np.cos(x_5[3,:]/2)
    normal_force_5 = l_5[0,:] + l_5[1,:]
    axs1[0].plot(t, x_5[1,:], linewidth= 2.5, label = '$\sigma = {f}$'.format(f = sigmas[2]))
    axs1[1].plot(t, foot_height_5,linewidth = 2.5)
    axs1[2].plot(t, normal_force_5, linewidth = 1.5)
    axs1[0].plot(t, x_ref[1,:], 'k-', linewidth= 2.5, label = 'Reference')
    axs1[1].plot(t, ref_foot_height, 'k-', linewidth = 2.5)
    axs1[2].plot(t, ref_normal_force, 'k-', linewidth = 1.5)
    axs1[0].set_title('ERM')
    axs1[0].set_ylabel('Base Height')
    axs1[0].legend()
    axs1[1].set_ylabel('Foot Height')
    axs1[2].set_ylabel('Normal Impulse')
    plt.show()

if __name__ == "__main__":
    x = np.loadtxt('data/single_legged_hopper/nominal_3/x_s.txt')
    u = np.loadtxt('data/single_legged_hopper/nominal_3/u_s.txt')
    l = np.loadtxt('data/single_legged_hopper/nominal_3/l_s.txt')
    t = np.loadtxt('data/single_legged_hopper/nominal_3/t.txt')
    plot(x,u, l, t)
    # sigmas = np.array([0.28, 0.50])
    # plot_CC()
    # plot_erm()

    # Visualization:
    # x = np.loadtxt('data/single_legged_hopper/erm_cc/x_0.4.txt')
    # t = t = np.loadtxt('data/single_legged_hopper/nominal/t1.txt')
    _file = "systems/urdf/single_legged_hopper.urdf"
    x = PiecewisePolynomial.FirstOrderHold(t, x)
    vis = Visualizer(_file)
    body_inds = vis.plant.GetBodyIndices(vis.model_index)
    base_frame = vis.plant.get_body(body_inds[0]).body_frame()
    vis.plant.WeldFrames(vis.plant.world_frame(), base_frame, RigidTransform())

    vis.visualize_trajectory(x)

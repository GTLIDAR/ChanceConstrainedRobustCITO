"""This is used for generateting visualization for the IEEE Access paper"""
import numpy as np
import matplotlib.pyplot as plt
from utilities import FindResource, GetKnotsFromTrajectory, load, generate_filename, generate_config
import systems.block.block as block
from pydrake.all import RigidTransform, PiecewisePolynomial
import concurrent.futures

def plot_control_trajectories(folder = None, name = None, sigmas = None):
    fig1, axs1 = plt.subplots(5,1)
    ref_soln = load('data/IEEE_Access/sliding_block/PaperResults/warm_start/warm_start.pkl')
    t = ref_soln['time'].reshape(101,1)
    ref_u = ref_soln['control'].reshape(101,1)
    axs1[0].plot(t, ref_u, label='Reference', linewidth=2.5)
    axs1[0].legend()
    for sigma in sigmas:
        # filename = f"{folder}/{name}_{sigma}"
        sigma_str = "{:.2e}".format(sigma)
        name = 'block' + '_erm' + '_sigma'+sigma_str+'.pkl'
        # name = f'block_cc_sigma{sigma}_beta0.65_theta0.65'
        filename=folder+'/'+name
        soln = load(filename)
        u = soln['control'].reshape(101,1)
        f = soln['force']
        axs1[1].plot(t[:], u[:], label=f'$\sigma$ ={sigma}', linewidth=2.5)
        axs1[2].plot(t[:], f[1,:]-f[3,:], label=f'$\sigma$ ={sigma}', linewidth=2.5)
        
    axs1[1].legend()
    axs1[1].spines["top"].set_visible(False)
    axs1[1].spines["right"].set_visible(False)
    axs1[2].spines["top"].set_visible(False)
    axs1[2].spines["right"].set_visible(False)
    # axs1[2].set_yticks([0,-2,-4,-6])
    plt.show()

def plot_traj(folder=None, config = None, ERM=False, CC=False):
    """config = [sigma, beth, theta]"""
    name = generate_filename(name="block", ERM=ERM, CC=CC, config=config)
    filename=folder+'/'+name
    soln = load(filename=filename)
    u = soln['control']
    f = soln['force']
    x = soln['state']
    t = np.linspace(start=0, stop=1, num=101)
    utraj = PiecewisePolynomial.FirstOrderHold(t, u)
    xtraj = PiecewisePolynomial.FirstOrderHold(t, x)
    ftraj = PiecewisePolynomial.FirstOrderHold(t, f)
    plant = block.Block()
    plant.plot_trajectories(xtraj=xtraj, utraj=utraj, ftraj=ftraj)

def plot_ERMCC_traj(config=None):
    plot_traj(folder="data/IEEE_Access/sliding_block/ERM+CC", config=config, ERM=True, CC=True)

def run_plot():
    configs = generate_config()
    for config in configs:
        print(f"sigma is{config[0]}; beta is {config[1]}; theta is {config[2]}")
        plot_ERMCC_traj(config=config)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     successes = executor.map(run_plot, configs)
        
def compare_traj():
    x_old= np.loadtxt('data/slidingblock/warm_start/x.txt')
    u_old = np.loadtxt('data/slidingblock/warm_start/u.txt')
    # u_init = u_init.reshape(trajopt.u.shape)
    l_old = np.loadtxt('data/slidingblock/warm_start/l.txt')
    soln = load("data/IEEE_Access/sliding_block/block_trajopt_nominal_tight.pkl")
    u_new = soln['control'].reshape(101,1)
    x_new = soln['state']
    l_new = soln['force']
    t = soln['time'].reshape(101,1)
    fig1, axs1 = plt.subplots(3,1)
    axs1[0].plot(t, u_old, label='old', linewidth=2.5)
    axs1[1].plot(t, x_old[0], label='old', linewidth=2.5)
    axs1[2].plot(t, l_old[1,:]-l_old[3,:], label='old', linewidth=2.5)
    fig2, axs2 = plt.subplots(3,1)
    axs2[0].plot(t, u_new, label='new', linewidth=2.5)
    axs2[1].plot(t, x_new[0], label='new', linewidth=2.5)
    axs2[2].plot(t, l_new[1,:]-l_new[3,:], label='new', linewidth=2.5)
    plt.show()

if __name__ == "__main__":
    run_plot()
    pass
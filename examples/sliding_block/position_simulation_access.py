"""
This script is used to generate position forward simulation for IEEE Access
"""
from systems.terrain import FlatTerrain
from os import stat_result
from systems.block.block import Block
import numpy as np
import matplotlib.pyplot as plt

import utilities as utils

# global parameters
capthick=4
linewidth=3.5
capsize=5
h = 0.01
frictions = np.array([ 0.3, 0.43, 0.57, 0.7])
target = 5
x0 = np.array([0, 0.5, 0, 0])

def run_simulation(ERM_dir=None, ERM_CC_dir=None):
    '''beta theta is fixed in this simulation'''
    reference_control = np.loadtxt('data/slidingblock/warm_start/u.txt')
    fig, axs = plt.subplots(3,1)
    sigma = 0.3
    for fric in frictions:
        # reference control
        plant = Block(terrain=FlatTerrain(friction=fric))
        plant.Finalize()
        t, x, f = plant.simulate(h, x0, u = reference_control.reshape(1,101), N = 101)
        y_bar = np.zeros(t.shape) + 5
        axs[0].plot(t, y_bar, 'k', linewidth =1)
        axs[0].plot(t, x[0, :], linewidth =3, label = f'$\mu$ = {fric}')
        # ERM control
        filename = f"{ERM_dir}/block_erm_{sigma}"
        soln = utils.load(filename)
        erm_control = soln['control'].reshape(101,1)
        t, x, f = plant.simulate(h, x0, u = erm_control.reshape(1,101), N = 101)
        axs[1].plot(t, y_bar, 'k', linewidth =1)
        axs[1].plot(t, x[0, :], linewidth =3, label = f'$\mu$ = {fric}')
        # ERM + CC control
        filename = f"{ERM_CC_dir}/block_erm_{sigma}"
        soln = utils.load(filename)
        erm_cc_control = soln['control'].reshape(101,1)
        t, x, f = plant.simulate(h, x0, u = erm_cc_control.reshape(1,101), N = 101)
        soln['control'].reshape(101,1)
        axs[2].plot(t, y_bar, 'k', linewidth =1)
        axs[2].plot(t, x[0, :], linewidth =3, label = f'$\mu$ = {fric}')

    for i in range(axs.shape[0]):
        axs[i].set_yticks([0, 2, 4, 6])
        axs[i].set_ylim([0,6.1])
        axs[i].set_xlim([0,1])
        # Hide the right and top spines
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].yaxis.set_ticks_position('left')
    plt.show()

def run_simulation_fixed_CC(ERM_dir= None, ERM_CC_dir=None):
    '''comparison between ERM and ERM+CC where beta theta parameters are fixed'''
    sigmas = np.array([0.01, 0.05, 0.1, 0.3, 1.0])
    # reference control
    reference_control = np.loadtxt('data/slidingblock/warm_start/u.txt')
    x = 0
    fig, axs = plt.subplots(1,1)
    final_positions=simulate_block_with_perturbed_frictions(frictions=frictions, control=reference_control)
    # position_errors = final_positions - 5
    # mean_position = np.mean(position_errors)
    mean_position, yerr = calculate_errors(final_positions=final_positions, target=target)
    axs.errorbar(x, mean_position, yerr=yerr, fmt='o', capsize=capsize,label='Reference',
                    elinewidth=linewidth, capthick=capthick)
    x=x+2
    for sigma in sigmas:
        # ERM control
        erm_control = load_ERM_control(sigma=sigma)
        final_positions = simulate_block_with_perturbed_frictions(frictions=frictions, control=erm_control)
        mean_position, yerr = calculate_errors(final_positions=final_positions, target=target)
        axs.errorbar(x, mean_position, yerr=yerr, fmt='o', capsize=capsize,label='ERM',
                        elinewidth=linewidth, capthick=capthick, color='darkblue')
        x = x+1
        # ERM + CC control
        erm_cc_control = load_ERM_CC_control(sigma=sigma)
        final_positions = simulate_block_with_perturbed_frictions(frictions=frictions, control=erm_cc_control)
        mean_position, yerr = calculate_errors(final_positions=final_positions, target=target)
        axs.errorbar(x, mean_position, yerr=yerr, fmt='o', capsize=capsize,label='ERM+CC',
                        elinewidth=linewidth, capthick=capthick, color='darkgreen')
        x = x+2
    plot_target_line(axs=axs)
    axs.legend()
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    plt.show()

def run_simulation_beta_theta(folder=None):
    '''beta theta varies in this simulation'''
    fig, axs = plt.subplots(1,1)
    betas = np.array([0.51, 0.6, 0.9])
    thetas = np.array([0.51, 0.6, 0.9])
    sigma=0.3
    x = 0
    # ERM control
    final_positions= np.zeros([4,1])
    i = 0
    erm_control = load_ERM_control(sigma=sigma)
    final_positions = simulate_block_with_perturbed_frictions(frictions=frictions, control=erm_control)
    mean_position, yerr = calculate_errors(final_positions=final_positions, target=target)
    axs.errorbar(x, mean_position, yerr=yerr, fmt='o', capsize=capsize,label='ERM',
                    elinewidth=linewidth, capthick=capthick, color='darkblue')
    x = x+2
    colors = ['green', 'purple', 'grey']
    # ERM + CC control
    for theta in thetas:
        j = 0
        for beta in betas:
            name =f"block_erm_cc_sigma{sigma}_beta{beta}_theta{theta}"
            filename = folder+'/'+ name
            soln = utils.load(filename=filename)
            control = soln['control'].reshape(101,1)
            final_positions= np.zeros([4,1])
            i=0
            for fric in frictions:
                plant = Block(terrain=FlatTerrain(friction=fric))
                plant.Finalize()
                t, state, f = plant.simulate(h, x0, u = control.reshape(1,101), N = 101)
                final_positions[i] = state[0,100]
                i = i+1
            position_errors = final_positions - 5
            mean_position = np.mean(position_errors)
            legend = f'$\ beta$={beta}, $\ theta$={theta}'
            yerr = np.zeros([2,1])
            yerr[0] = np.abs(position_errors.min()-mean_position)
            yerr[1] = np.abs(position_errors.max()-mean_position)
            axs.errorbar(x, mean_position, yerr=yerr, color = colors[j],
                             fmt='o', capsize=4,label=legend, elinewidth=linewidth, capthick=capthick)
            j=j+1
            x=x+1
        x=x+1
    axs.set_xticks([])
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    plot_target_line(axs=axs)
    axs.legend()
    plt.show()

def simulate_block_with_perturbed_frictions(frictions=0.5, control=None):
    final_positions= np.zeros([4,1])
    i=0
    for fric in frictions:
        plant = Block(terrain=FlatTerrain(friction=fric))
        plant.Finalize()
        t, state, f = plant.simulate(h, x0, u = control.reshape(1,101), N = 101)
        final_positions[i] = state[0,100]
        i = i+1
    return final_positions

def load_ERM_control(sigma=None):
    ERM_dir = "data/IEEE_Access/sliding_block/ERM"
    filename = f"{ERM_dir}/block_erm_{sigma}"
    soln = utils.load(filename)
    control = soln['control'].reshape(101,1)
    return control

def load_ERM_CC_control(sigma=None):
    ERM_CC_dir = "data/IEEE_Access/sliding_block/ERM_CC"
    filename = f"{ERM_CC_dir}/block_erm_{sigma}"
    soln = utils.load(filename)
    control = soln['control'].reshape(101,1)
    return control

def calculate_errors(final_positions=None, target=None):
    position_errors = final_positions - target
    mean_position = np.mean(position_errors)
    yerr = np.zeros([2,1])
    yerr[0] = np.abs(position_errors.min()-mean_position)
    yerr[1] = np.abs(position_errors.max()-mean_position)
    return mean_position, yerr

def plot_target_line(axs=None):
    target_line =  np.arange(-1, 15)
    y = np.zeros(target_line.shape)
    axs.plot(target_line, y, 'k--')

if __name__ == "__main__":
    ERM_folder = "data/IEEE_Access/sliding_block/ERM"
    ERM_CC_folder = "data/IEEE_Access/sliding_block/ERM_CC"
    # run_simulation(ERM_dir=ERM_folder, ERM_CC_dir=ERM_CC_folder)
    run_simulation_beta_theta(folder="data/IEEE_Access/sliding_block/ERM_CC_Beta_theta_scaleOption2_tight")
    # run_simulation_fixed_CC()
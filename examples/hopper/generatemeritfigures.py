"""
Script for generating merit score figures

"""
import os, csv
import numpy as np
import matplotlib.pyplot as plt
import decorators as deco
import utilities as utils

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


def load_data_recursive(directory, filename = 'trajoptresults.pkl'):
    data = [utils.load(os.path.join(file, filename)) for file in utils.find_filepath_recursive(directory, filename)]
    return data

def get_nominal_data(directory, scorename):
    filename = os.path.join(directory, 'trajoptresults.pkl')
    data = utils.load(filename)  
    return data['merit_score'][scorename], data['sensitivity_score'][scorename][scorename]

def get_erm_data(directory, scorename):
    dataset = load_data_recursive(directory)
    meritscores = np.full((1,len(dataset)), np.nan)
    sensitivities = np.full((1,len(dataset)), np.nan)
    sigmas = np.zeros((1,len(dataset)))
    for n in range(len(dataset)):
        sigmas[:,n] = dataset[n]['sigma']
        if dataset[n]['success']:
            meritscores[:, n] = dataset[n]['merit_score'][scorename]
            sensitivities[:, n] = dataset[n]['sensitivity_score'][scorename][scorename]
        else:
            meritscores[:, n] = float('nan')
            sensitivities[:, n] = float('nan')
    # Sort the values
    sort_idx = np.argsort(sigmas)
    sigmas = sigmas[:, sort_idx[0,:]]
    meritscores = meritscores[:, sort_idx[0,:]]
    sensitivities = sensitivities[:, sort_idx[0,:]]
    return  meritscores, sensitivities, sigmas

def get_chance_data(directory, scorename):
    # Load the data and filter out only the cases for which beta and theta are the same
    dataset = [data for data in load_data_recursive(directory) if data['theta'] == data['beta']]
    # Get the unique theta and sigma values
    sigmas = list(set([data['sigma'] for data in dataset]))
    thetas = list(set([data['theta'] for data in dataset]))
    #Sort them
    sigmas = sorted(sigmas)
    thetas = sorted(thetas)
    meritscores = np.full((len(thetas), len(sigmas)), np.nan)
    sensitivities = np.full((len(thetas), len(sigmas)), np.nan)
    for data in dataset:
        sindex = sigmas.index(data['sigma'])
        tindex = thetas.index(data['theta'])
        if data['success']:
            meritscores[tindex, sindex] = data['merit_score'][scorename]
            sensitivities[tindex, sindex] = data['sensitivity_score'][scorename][scorename]
    labels = [f'$\\theta$={theta:0.1f}, $\\beta$={theta:0.1f}' for theta in thetas]    
    return meritscores, sensitivities, labels

def get_chance_data_hopper(directory, scorename):
    # Load the data and filter out only the cases for which beta and theta are the same
    dataset = [data for data in load_data_recursive(directory) if data['beta'] == 0.5]
    # Get the unique theta and sigma values
    sigmas = list(set([data['sigma'] for data in dataset]))
    thetas = list(set([data['theta'] for data in dataset]))
    #Sort them
    sigmas = sorted(sigmas)
    thetas = sorted(thetas)
    meritscores = np.full((len(thetas), len(sigmas)), np.nan)
    sensitivities = np.full((len(thetas), len(sigmas)), np.nan)
    for data in dataset:
        sindex = sigmas.index(data['sigma'])
        tindex = thetas.index(data['theta'])
        if data['success']:
            meritscores[tindex, sindex] = data['merit_score'][scorename]
            sensitivities[tindex, sindex] = data['sensitivity_score'][scorename][scorename]
        else:
            meritscores[tindex, sindex] = float('nan')
            sensitivities[tindex, sindex] = float('nan')
    labels = [f'$\\theta$={theta:0.1f}, $\\beta$=0.5' for theta in thetas]    
    return meritscores, sensitivities, labels, sigmas

@deco.showable_fig
@deco.saveable_fig
def plot_sensitivities(sigmas, data, nominal, labels):
    """Make a plot comparing the sensitivity scores"""
    fig, axs = plt.subplots(1,1)
    # Add in the nominal data line
    ref = nominal*np.ones_like(sigmas)
    axs.plot(sigmas[0,:], ref[0,:], label="Reference", linewidth=1.5, linestyle='-')   
    for n in range(data.shape[0]):
        axs.plot(sigmas[0,:], data[n,:], label=labels[n], linewidth=1.5, marker='o')
    axs.set_yscale('symlog', linthreshy=1e-7)
    axs.set_xscale('log')
    axs.grid(True)
    axs.set_xlabel('Uncertainty ($\sigma$)')
    axs.set_ylabel('Sensitivity Score')
    axs.legend()
    return fig, axs

@deco.showable_fig
@deco.saveable_fig
def plot_infeasibilities(sigmas, data, nominal, labels):
    """Make a plot comparing the infeasibility scores"""
    fig, axs = plt.subplots(1,1)
    # Add in the nominal data line
    ref = nominal*np.ones_like(sigmas)
    axs.plot(sigmas[0,:], ref[0,:], label="Reference", linewidth=1.5, linestyle='--')
    for n in range(data.shape[0]):
        axs.plot(sigmas[0,:], data[n,:], label=labels[n], linewidth=1.5, marker='o')
    axs.set_yscale('symlog', linthreshy=1e-7)
    axs.set_xscale('log')
    axs.grid(True)
    axs.set_xlabel('Uncertainty ($\sigma$)')
    axs.set_ylabel('Merit Score')
    axs.legend()
    return fig, axs

def combine_erm_cc_data(ermMerit, ermSens, ermSig, ccMerit, ccSens, ccSig):
    # Find the sigmas in ccSig that are also in ermSig
    both = set(ermSig).intersection(ccSig)
    cc_idx = sorted([ccSig.index(x) for x in both])
    erm_idx = sorted([ermSig.index(x) for x in both])
    common_sigma = [ermSig[i] for i in erm_idx]
    # Filter down and combine with the ERM data
    merit = np.concatenate([ermMerit[:, erm_idx], ccMerit[:, cc_idx]], axis=0)
    sens = np.concatenate([ermSens[:, erm_idx], ccSens[:, cc_idx]], axis=0)
    return merit, sens, common_sigma

def compare_hopper_merit(nominal_dir, erm_dir, erm_cc_dir, outdir):
    # The directories

    # Load the data
    scorename = 'normal_distance'
    nominal_merit, nominal_sensitivity = get_nominal_data(nominal_dir, scorename)
    erm_merit, erm_sensitivity, sigmas = get_erm_data(erm_dir, scorename)
    cc_merit, cc_sensitivity, labels, cc_sigmas = get_chance_data_hopper(erm_cc_dir, scorename)
    # Combine the data
    sigmas = sigmas.tolist()[0]
    merit_data, sensitivity_data, common_sigma = combine_erm_cc_data(erm_merit, erm_sensitivity, sigmas, cc_merit, cc_sensitivity, cc_sigmas)
    #merit_data = np.concatenate([erm_merit, cc_merit], axis=0)
    #sensitivity_data = np.concatenate([erm_sensitivity, cc_sensitivity], axis=0)
    common_sigma = np.array([common_sigma])
    labels.insert(0, 'ERM')
    
    plot_infeasibilities(common_sigma, merit_data, nominal_merit, labels, show=False, savename=os.path.join(outdir, 'HopperMerit.png'))
    plot_sensitivities(common_sigma, sensitivity_data, nominal_sensitivity, labels, show=False, savename=os.path.join(outdir,'HopperSensitivity.png'))
    print(f'Finished. Plot saved at {outdir}')

def save_merit_to_table(nominal_dir, erm_dir, erm_cc_dir, outdir):
    # Load the data
    scorename = 'normal_distance'
    nominal_merit, nominal_sensitivity = get_nominal_data(nominal_dir, scorename)
    erm_merit, erm_sensitivity, sigmas = get_erm_data(erm_dir, scorename)
    cc_merit, cc_sensitivity, labels, cc_sigmas = get_chance_data_hopper(erm_cc_dir, scorename)
    # Combine the data
    sigmas = sigmas.tolist()[0]
    merit_data, sensitivity_data, common_sigma = combine_erm_cc_data(erm_merit, erm_sensitivity, sigmas, cc_merit, cc_sensitivity, cc_sigmas)

    sens_file = os.path.join(outdir,'Sensitivity.csv')
    common_sigma.insert(0, 'Sigma')
    labels.insert(0, "ERM")
    # Copy the nominal merit
    nominal_merit = [nominal_merit]*merit_data.shape[1]
    nominal_sensitivity = [nominal_sensitivity]*sensitivity_data.shape[1]

    # Write the merit data
    merit_file = os.path.join(outdir, 'Merit.csv')
    with open(merit_file, 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        #Write the header
        writer.writerow(common_sigma)
        # Write the nominal data
        nominal_merit.insert(0, 'Reference')
        writer.writerow(nominal_merit)
        # Write the remaining rows
        for n in range(merit_data.shape[0]):
            data = merit_data[n].tolist()
            data.insert(0, labels[n])
            writer.writerow(data)
    print(f"Merit data saved to {merit_file}")

    # Write the sensitivity data
    with open(sens_file, 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(sigmas)
        # Write nominal data
        nominal_sensitivity.insert(0, 'Reference')
        writer.writerow(nominal_sensitivity)
        # Write remaining data
        for n in range(sensitivity_data.shape[0]):
            data = sensitivity_data[n].tolist()
            data.insert(0, labels[n])
            writer.writerow(data)
    print(f"Sensitivity data saved to {sens_file}")




if __name__ == "__main__":
    nominal_dir = os.path.join('examples','hopper','reference_linear','strict_nocost')
    erm_dir = os.path.join('examples','hopper','robust_nonlinear','erm_1e5','success')
    erm_cc_dir = os.path.join('examples','hopper','robust_nonlinear','cc_erm_1e5_mod','success')
    outdir = os.path.join('examples','hopper','robust_nonlinear','cc_erm_1e5_mod','success')
    compare_hopper_merit(nominal_dir, erm_dir, erm_cc_dir, outdir)
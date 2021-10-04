"""
Script for generating merit score figures

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import decorators as deco
import utilities as utils

def load_data_recursive(directory, filename = 'trajoptresults.pkl'):
    data = [utils.load(os.path.join(file, filename)) for file in utils.find_filepath_recursive(directory, filename)]
    return data

def get_nominal_data(directory, scorename):
    filename = os.path.join(directory, 'trajoptresults.pkl')
    data = utils.load(filename)  
    return data['merit_score'][scorename], data['sensitivity_score'][scorename][scorename]

def get_erm_data(directory, scorename):
    dataset = load_data_recursive(directory)
    meritscores = np.zeros((1,len(dataset)))
    sensitivities = np.zeros((1,len(dataset)))
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
    meritscores = np.zeros((len(thetas), len(sigmas)))
    sensitivities = np.zeros((len(thetas), len(sigmas)))
    for data in dataset:
        sindex = sigmas.index(data['sigma'])
        tindex = thetas.index(data['theta'])
        if data['success']:
            meritscores[tindex, sindex] = data['merit_score'][scorename]
            sensitivities[tindex, sindex] = data['sensitivity_score'][scorename][scorename]
        else:
            meritscores[tindex, sindex] = float('nan')
            sensitivities[tindex, sindex] = float('nan')
    labels = [f'\\theta={theta:0.1f}, \\beta={theta:0.1f}' for theta in thetas]    
    return meritscores, sensitivities, labels

@deco.showable_fig
@deco.saveable_fig
def plot_sensitivities(sigmas, data, nominal, labels):
    """Make a plot comparing the sensitivity scores"""
    fig, axs = plt.subplots(1,1)
    for n in range(data.shape[0]):
        axs.plot(sigmas[0,:], data[n,:], label=labels[n], linewidth=1.5, marker='o')
    axs.set_yscale('symlog', linthreshy=1e-7)
    axs.set_xscale('log')
    axs.grid(True)
    # Add in the nominal data line
    ref = nominal*np.ones_like(sigmas)
    axs.plot(sigmas[0,:], ref[0,:], label="Reference", linewidth=1.5, linestyle='--')   
    axs.set_xlabel('Uncertainty (\sigma)')
    axs.set_ylabel('Sensitivity Score')
    axs.legend()
    return fig, axs

@deco.showable_fig
@deco.saveable_fig
def plot_infeasibilities(sigmas, data, nominal, labels):
    """Make a plot comparing the infeasibility scores"""
    fig, axs = plt.subplots(1,1)
    for n in range(data.shape[0]):
        axs.plot(sigmas[0,:], data[n,:], label=labels[n], linewidth=1.5, marker='o')
    axs.set_yscale('symlog', linthreshy=1e-7)
    axs.set_xscale('log')
    axs.grid(True)
    # Add in the nominal data line
    ref = nominal*np.ones_like(sigmas)
    axs.plot(sigmas[0,:], ref[0,:], label="Reference", linewidth=1.5, linestyle='--')
    axs.set_xlabel('Uncertainty (\sigma)')
    axs.set_ylabel('Merit Score')
    axs.legend()
    return fig, axs

def main_block():
    # The directories
    nominal_dir = os.path.join("data","IEEE_Access","sliding_block","PaperResults","warm_start","reference")
    erm_dir = os.path.join("data","IEEE_Access","sliding_block","ERM")
    erm_cc_dir = os.path.join("data","IEEE_Access","sliding_block","ERM+CC")
    # Load the data
    scorename = 'friction_cone'
    nominal_merit, nominal_sensitivity = get_nominal_data(nominal_dir, scorename)
    erm_merit, erm_sensitivity, sigmas = get_erm_data(erm_dir, scorename)
    cc_merit, cc_sensitivity, labels = get_chance_data(erm_cc_dir, scorename)
    merit_data = np.concatenate([erm_merit, cc_merit], axis=0)
    sensitivity_data = np.concatenate([erm_sensitivity, cc_sensitivity], axis=0)
    labels.insert(0, 'ERM')
    outdir = os.path.join('examples','robust_block')
    plot_infeasibilities(sigmas, merit_data, nominal_merit, labels, show=False, savename=os.path.join(outdir, 'BlockMerit.png'))
    plot_sensitivities(sigmas, sensitivity_data, nominal_sensitivity, labels, show=False, savename=os.path.join(outdir,'BlockSensitivity.png'))
    print(f'Finished. Plot saved at {outdir}')

if __name__ == "__main__":
    main_block()
import numpy as np
from numpy.lib import utils
from trajopt.contactimplicit import ContactConstraintViewer as CCV
from trajopt.contactimplicit import ContactImplicitDirectTranscription
from systems.block.block import Block
from blockOpt_access import *
import utilities as utils
import matplotlib.pyplot as plt
from systems.terrain import FlatTerrain
'''
This script calculates merit score for friction and normal distance constraint violations
'''
# global parameters
TRAJOPT = setup_nominal_block_trajopt()
markersize = 10
linewidth = 3
sigmas = np.array([0.01, 0.05, 0.1, 0.3, 1.0])
ERM_folder = "data/IEEE_Access/sliding_block/ERM"
ERM_CC_folder = "data/IEEE_Access/sliding_block/ERM+CC"
# ERM_folder = "data/IEEE_Access/sliding_block/PaperResults/ERM"
# ERM_CC_folder = "data/IEEE_Access/sliding_block/PaperResults/ERM+CC"
reference = 'data/IEEE_Access/sliding_block/PaperResults/warm_start/warm_start.pkl'
betas = np.array([0.51, 0.60, 0.7, 0.8, 0.9])
thetas = np.array([0.51, 0.60, 0.7, 0.8, 0.9])

def main():
    trajopt = setup_nominal_block_trajopt()
    filename = 'data/IEEE_Access/sliding_block/block_trajopt.pkl'
    filename = 'data/IEEE_Access/sliding_block/ERM/block_erm_1.0'
    soln = utils.load(filename)
    constraintViewer = CCV(trajopt=trajopt, result_dict=soln)
    cstr_dict = constraintViewer.calc_constraint_values()
    merit_score = calculate_merit_score(cstr_dict['friction_cone'])
    print(merit_score)

def calculate_merit_score(cstr=None, key=None):
    '''This method calculates the merit score given the constraints'''
    merit_score = 0
    if key == 'dynamics' or key == 'TimestepConstraint':
        for i in range(cstr.shape[0]):
            merit_score = merit_score + np.sum(np.square(cstr[i,:]))
    elif key == 'normal_distance' or key == 'sliding_velocity' or key == 'friction_cone':
        row, col = cstr.shape
        for i in range(row):
            # inequality constraint
            if i < row*2/3:
                merit_score = merit_score + np.sum(np.square(np.minimum(0, cstr[i,:])))
            # equality constraint
            else:
                merit_score = merit_score + np.sum(np.square(cstr[i,:]))
    else:
        print('INVALID KEY')
        print(key)
    return merit_score

def perturbe_terrain_friction(eps=None):
    plant = Block(terrain=FlatTerrain(friction=0.5+eps))
    plant.Finalize()
    context = plant.multibody.CreateDefaultContext()
    trajopt = ContactImplicitDirectTranscription(plant=plant,
                                                context=context,
                                                num_time_samples=101,
                                                maximum_timestep=0.01,
                                                minimum_timestep=0.01)
    return trajopt

def calculate_robustness(filename=None):
    '''This method calculates the derivative of the merit score through central difference method'''
    eps=1e-10
    trajopt_1 = perturbe_terrain_friction(eps)
    trajopt_2 = perturbe_terrain_friction(-eps)
    cstr_1 = get_constraints(trajopt=trajopt_1, filename=filename)
    cstr_2 = get_constraints(trajopt=trajopt_2, filename=filename)
    merit_score_1 = calculate_merit_score(cstr=cstr_1['friction_cone'], key='friction_cone')
    merit_score_2 = calculate_merit_score(cstr=cstr_2['friction_cone'], key='friction_cone')
    robustness = (merit_score_1-merit_score_2)/(2*eps)
    return robustness
    # return np.abs(robustness)

def robustness_comparision():
    '''This method generates the robustness comparision figure'''
    fig, axs = plt.subplots(1,1)
    ref_robustness = np.zeros(sigmas.shape)
    erm_robustness = np.zeros(sigmas.shape)
    # make the dictionary to score merit scores
    erm_cc_robustness = {}
    for beta in betas:
        for theta in thetas:
            erm_cc_robustness[f'beta{beta}theta{theta}'] = np.zeros(sigmas.shape)
    x = np.arange(len(sigmas))
    i = 0
    for sigma in sigmas:
        sigma_str = "{:.2e}".format(sigma)
        ERM_filename = ERM_folder+'/'+'block_erm_sigma'+sigma_str+'.pkl'
        ref_robustness[i] = calculate_robustness(filename=reference)
        erm_robustness[i] = calculate_robustness(filename=ERM_filename)
        for beta in betas:
            for theta in thetas:
                beta_str = "{:.2e}".format(beta)
                theta_str = "{:.2e}".format(theta)
                erm_cc_filename = ERM_CC_folder+'/'+'block_erm'+'_sigma'+sigma_str+'_beta'+beta_str+'_theta'+theta_str+'.pkl'
                erm_cc_cstr = get_constraints(filename=erm_cc_filename)
                erm_cc_robustness[f'beta{beta}theta{theta}'][i] = calculate_robustness(filename=erm_cc_filename)
        i=i+1
    for beta in betas:
        for theta in thetas:
            axs.plot(x, erm_cc_robustness[f'beta{beta}theta{theta}'], '-^', markersize=markersize, linewidth=linewidth, label=f'beta{beta}theta{theta}')
    axs.plot(x, erm_robustness, '-bs', markersize = markersize,linewidth=linewidth, label = 'ERM')
    axs.plot(x, ref_robustness, '-ko', markersize = markersize,linewidth=linewidth, label = 'Reference')
    axs.set_yscale('symlog')
    plt.xticks(x, tuple(sigmas))
    axs.set_xlabel('$\sigma$')
    axs.set_ylabel('Robustness Score')
    axs.set_title('Robustness Score')
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.legend()
    plt.show()

def log_bar_graph_block_total():
    '''
    This method is used to generate the log scale bar graph including the 
    '''
    width=0.3
    erm_filename = 'data/IEEE_Access/sliding_block/ERM_CC/block_erm_1.0'
    erm_cc_filename = ERM_CC_folder + '/' + 'block_cc_sigma1.0_beta0.65_theta0.65'
    ref_dict = get_constraints(filename=reference)
    erm_cstr_dict = get_constraints(filename=erm_filename)
    erm_cc_cstr_dict = get_constraints(filename=erm_cc_filename)
    i=1
    erm_merit_scores = []
    erm_cc_merit_scores = []
    ref_merit_scores = []
    x = np.arange(0, 5)
    fig, axs = plt.subplots(1,1)
    keys = []
    for key in erm_cstr_dict:
        if key != 'TimestepConstraint':
            keys.append(key)
            ref_merit_scores.append(calculate_merit_score(ref_dict[key], key=key))
            erm_merit_scores.append(calculate_merit_score(erm_cstr_dict[key], key=key))
            erm_cc_merit_scores.append(calculate_merit_score(erm_cc_cstr_dict[key], key=key))
            i = i+1
    keys.append('Total')
    # total_merit_score = np.sum(ref_merit_scores)
    ref_merit_scores.append(np.sum(ref_merit_scores))
    erm_merit_scores.append(np.sum(erm_merit_scores))
    erm_cc_merit_scores.append(np.sum(erm_cc_merit_scores))
    axs.bar(x-width/2, tuple(ref_merit_scores), width=width, label='Reference', color='k', bottom=1e-12)
    axs.bar(x+width/2, tuple(erm_merit_scores), width=width, label='ERM', color='b', bottom=1e-12)
    axs.bar(x+width*3/2, tuple(erm_cc_merit_scores), width=width, label='ERM+CC', color='g', bottom=1e-12)
    plt.xticks(x+width/2, tuple(keys))

    # plt.ylim(1e-15, 1e5)
    axs.set_yscale('log')
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.set_title('Merit Score comparision across different constraints, $\sigma$ = 1.0')
    plt.legend()
    plt.show()
    # plt.plot(x, total_merit_score)
    
def merit_score_comparison():
    '''
    This method generates the figure for merit score comparison btween nominal, ERM and ERM+CC for
    only the friction cone constraint violations. 
    '''
    ref_merit_scores = np.zeros(sigmas.shape)
    erm_merit_scores = np.zeros(sigmas.shape)
    # make the dictionary to score merit scores
    erm_cc_scores = {}
    for beta in betas:
        for theta in thetas:
            erm_cc_scores[f'beta{beta}theta{theta}'] = np.zeros(sigmas.shape)
    x = np.arange(len(sigmas))
    fig, axs = plt.subplots(1,1)
    i = 0
    for sigma in sigmas:
        sigma_str = "{:.2e}".format(sigma)
        ERM_filename = ERM_folder+'/'+'block_erm_sigma'+sigma_str+'.pkl'
        ref_cstr = get_constraints(filename=reference)
        ERM_cstr = get_constraints(filename=ERM_filename)
        ref_merit_scores[i] = calculate_merit_score(ref_cstr['friction_cone'], key='friction_cone')
        erm_merit_scores[i] = calculate_merit_score(ERM_cstr['friction_cone'], key='friction_cone')
        for beta in betas:
            for theta in thetas:
                beta_str = "{:.2e}".format(beta)
                theta_str = "{:.2e}".format(theta)
                # erm_cc_scores['']
                erm_cc_filename = ERM_CC_folder+'/'+'block_erm'+'_sigma'+sigma_str+'_beta'+beta_str+'_theta'+theta_str+'.pkl'
                erm_cc_cstr = get_constraints(filename=erm_cc_filename)
                erm_cc_scores[f'beta{beta}theta{theta}'][i] = calculate_merit_score(erm_cc_cstr['friction_cone'], key='friction_cone')
        i=i+1
    for beta in betas:
        for theta in thetas:
            axs.plot(x, erm_cc_scores[f'beta{beta}theta{theta}'], '-^', markersize=markersize, linewidth=linewidth, label=f'beta{beta}theta{theta}')
    axs.plot(x, erm_merit_scores, '-bs', markersize = markersize,linewidth=linewidth, label = 'ERM')
    axs.plot(x, ref_merit_scores, '-ko', markersize = markersize,linewidth=linewidth, label = 'Reference')
    axs.set_yscale('symlog')
    plt.xticks(x, tuple(sigmas))
    axs.set_xlabel('$\sigma$')
    axs.set_ylabel('Merit Score')
    axs.set_title('Merit Score')
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.legend()
    plt.show()
    
def get_constraints(trajopt=TRAJOPT, filename=None):
    soln = utils.load(filename)
    constraintViewer = CCV(trajopt=trajopt, result_dict=soln)
    cstr_dict = constraintViewer.calc_constraint_values()
    return cstr_dict

if __name__ == "__main__":
    # main()
    # log_bar_graph_block_total()
    merit_score_comparison()
    robustness_comparision()
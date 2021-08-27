import numpy as np
from numpy.lib import utils
from trajopt.contactimplicit import ContactConstraintViewer as CCV
from trajopt.contactimplicit import ContactImplicitDirectTranscription
from systems.block.block import Block
from blockOpt_access import setup_nominal_block_trajopt
import utilities as utils
import matplotlib.pyplot as plt
from systems.terrain import FlatTerrain
'''
This script calculates merit score for friction and normal distance constraint violations
'''
# global parameters
trajopt = setup_nominal_block_trajopt()
markersize = 10
linewidth = 3

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
            merit_score = merit_score + np.sum(np.square(cstr[i,:-1]))
    elif key == 'normal_distance' or key == 'sliding_velocity' or key == 'friction_cone':
        row, col = cstr.shape
        for i in range(row):
            # inequality constraint
            if i < row*2/3:
                merit_score = merit_score + np.sum(np.square(np.minimum(0, cstr[i,:-1])))
            # equality constraint
            else:
                merit_score = merit_score + np.sum(np.square(cstr[i,:-1]))
    else:
        print('INVALID KEY')
        print(key)
    return merit_score

def perturbe_terrain_height(eps=None):
    plant = Block(terrain=FlatTerrain(height=eps))
    plant.Finalize()
    context = plant.multibody.CreateDefaultContext()
    trajopt = ContactImplicitDirectTranscription(plant=plant,
                                                context=context)
    
    return trajopt

def log_bar_graph_block_total():
    '''
    This method is used to generate the log scale bar graph including the 
    '''
    filename = 'data/IEEE_Access/sliding_block/ERM/block_erm_1.0'
    cstr_dict = get_constraints(filename=filename)
    i=0
    # merit_scores = np.zeros([len(cstr_dict), 1])
    merit_scores = []
    x = np.arange(0, len(cstr_dict))
    fig, axs = plt.subplots(1,1)
    keys = []
    for key in cstr_dict:
        if key != 'TimestepConstraint':
            keys.append(key)
            merit_scores.append(calculate_merit_score(cstr_dict[key], key=key))
        # i = i+1
    keys.append('Total')
    total_merit_score = np.sum(merit_scores)
    merit_scores.append(total_merit_score)
    # print(keys)
    axs.bar(keys, merit_scores, alpha = 0.5)
    # plt.ylim(1e-15, 1e5)
    axs.set_yscale('log')
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    plt.show()
    # plt.plot(x, total_merit_score)
    
def merit_score_comparison():
    '''
    This method generates the figure for merit score comparison btween nominal, ERM and ERM+CC for
    only the friction cone constraint violations. 
    '''
    sigmas = np.array([0.01, 0.05, 0.1, 0.3, 1.0])
    ERM_folder = "data/IEEE_Access/sliding_block/ERM_tight"
    ERM_CC_folder = "data/IEEE_Access/sliding_block/ERM_CC"
    erm_merit_scores = np.zeros(sigmas.shape)
    erm_cc_merit_scores = np.zeros(sigmas.shape)
    fig, axs = plt.subplots(1,1)
    i = 0
    for sigma in sigmas:
        ERM_filename = ERM_folder + '/'+ f'block_erm_{sigma}'
        ERM_CC_filename = ERM_CC_folder + '/'+ f'block_erm_{sigma}'
        ERM_cstr = get_constraints(filename=ERM_filename)
        ERM_CC_cstr = get_constraints(filename=ERM_CC_filename)
        erm_merit_scores[i] = calculate_merit_score(ERM_cstr['friction_cone'], key='friction_cone')
        erm_cc_merit_scores[i] = calculate_merit_score(ERM_CC_cstr['friction_cone'], key='friction_cone')
        i=i+1
    x = np.arange(len(sigmas))
    axs.plot(x, erm_merit_scores, '-bs', markersize = markersize,linewidth=linewidth, label = 'ERM')
    axs.plot(x, erm_cc_merit_scores, '-g^', markersize = markersize,linewidth=linewidth, label = 'ERM+CC')
    axs.set_yscale('log')
    axs.set_xticks([])
    axs.legend()
    plt.show()
    
def get_constraints(filename=None):
    soln = utils.load(filename)
    constraintViewer = CCV(trajopt=trajopt, result_dict=soln)
    cstr_dict = constraintViewer.calc_constraint_values()
    return cstr_dict

if __name__ == "__main__":
    # main()
    log_bar_graph_block_total()
    merit_score_comparison()
"""
Contact Implicit Trajectory Optimization for a sliding block. 
This script is used mainly to test the implementation of Contact Implicit Trajectory Optimization in contactimplicit.py

The goal is to move a 1kg block 5m in 1s. The timesteps are fixed, and the objective is to minimize the control cost and the state deviation from the final position. Boundary constraints are added to the problem to ensure the block starts and stops at rest and at the desired positions. In this example, the timesteps are fixed and equal. 

Luke Drnach
October 15, 2020
John Zhang
August, 20, 2021
"""
# Imports
from os import name
import timeit
import numpy as np
import matplotlib.pyplot as plt
from trajopt.contactimplicit import ContactImplicitDirectTranscription, OptimizationOptions
from trajopt.robustContactImplicit import OptimizationOptions, ChanceConstrainedContactImplicit
from trajopt.constraints import NCCImplementation, NCCSlackType
from systems.block.block import Block
from pydrake.solvers.snopt import SnoptSolver
import utilities as utils
from IEEE_figures import plot_multi_trajectories
import concurrent.futures

def run_block_trajopt_ERM(friction_multipler = 1e6, scale_option=1):
    friction_sigmas = np.array([0.1, 0.3, 1])
    # friction_sigmas = np.array([ 0.3])
    friction_bias = 0.01
    folder = f"data/IEEE_Access/sliding_block/ERM_{friction_multipler}_scaleOption_{scale_option}"
    distance_variance = 0.1
    distance_multiplier = 1e6
    distance_erm_params = np.array([distance_variance, distance_multiplier])
    for sigma in friction_sigmas:
        friction_erm_params = np.array([sigma, friction_bias, friction_multipler])
        print(f"Friction Variance is {sigma}")
        name = f"block_erm_{sigma}"
        run_block_trajopt(friction_erm_params=friction_erm_params, 
                            distance_erm_params=distance_erm_params,
                            uncertainty_option=3, cc_option=1, save_folder=folder, 
                            save_name=name, scale_option=scale_option)
    plot_multi_trajectories(folder=folder, name='block_erm', sigmas=friction_sigmas)

def run_opt_erm(sigma):
    friction_multipler = 5*1e4
    friction_bias = 0.01
    distance_variance = 0.1
    distance_multiplier = 1e6
    distance_erm_params = np.array([distance_variance, distance_multiplier])
    friction_erm_params = np.array([sigma, friction_bias, friction_multipler])
    print(type(sigma))
    print(f"Friction Variance is {sigma}")
    # name = f"block_erm_cc_sigma{sigma}_beta{beta}_theta{theta}"
    name = 'block'
    sigma_str = "{:.2e}".format(sigma)
    name = name + '_erm'
    name=name+'_sigma'+sigma_str
    cc_params=[0.65, 0.65, sigma]
    run_block_trajopt(friction_erm_params=friction_erm_params, cc_params=cc_params,
                        distance_erm_params=distance_erm_params,
                        uncertainty_option=3, cc_option=1, save_folder="data/IEEE_Access/sliding_block/ERM", 
                        save_name=name, scale_option=1, tight=True)
    
def run_opt_erm_cc(configs):
    sigma = configs[0]
    beta = configs[1]
    theta = configs[2]
    friction_multipler = 5*1e4
    friction_bias = 0.01
    distance_variance = 0.1
    distance_multiplier = 1e6
    distance_erm_params = np.array([distance_variance, distance_multiplier])
    friction_erm_params = np.array([sigma, friction_bias, friction_multipler])
    print(f"Friction Variance is {sigma}")
    # name = f"block_erm_cc_sigma{sigma}_beta{beta}_theta{theta}"
    name = 'block'
    sigma_str = "{:.2e}".format(sigma)
    beta_str = "{:.2e}".format(beta)
    theta_str = "{:.2e}".format(theta)
    name = name + '_erm'
    name=name+'_sigma'+sigma_str+'_beta'+beta_str+'_theta'+theta_str
    cc_params=[beta, theta, sigma]
    run_block_trajopt(friction_erm_params=friction_erm_params, cc_params=cc_params,
                        distance_erm_params=distance_erm_params,
                        uncertainty_option=3, cc_option=3, save_folder="data/IEEE_Access/sliding_block/ERM+CC", 
                        save_name=name, scale_option=1, tight=True)

def run_block_trajopt_ERM_CC(folder=None, scale_option=1, sigma = 0.01, beta = 0.6, theta=0.6, tight=False,
                            ermOption=1, ccOption=1):
    friction_sigmas = np.array([0.01, 0.05, 0.1, 0.3, 1])
    # friction_sigmas = np.array([ 0.01])
    friction_multipler = 1e6
    friction_bias = 0.01
    # folder = folder + f"_scaleOption2"
    if tight is True:
        folder = folder+"_tight"
    distance_variance = 0.1
    distance_multiplier = 1e6
    distance_erm_params = np.array([distance_variance, distance_multiplier])
    for sigma in friction_sigmas:
        friction_erm_params = np.array([sigma, friction_bias, friction_multipler])
        print(f"Friction Variance is {sigma}")
        # name = f"block_erm_cc_sigma{sigma}_beta{beta}_theta{theta}"
        name = 'block'
        sigma_str = "{:.2e}".format(sigma)
        if ermOption is not 1:
            name = name + '_erm'
        if ccOption is not 1:
            name = name + '_cc'
            beta_str = "{:.2e}".format(beta)
            theta_str = "{:.2e}".format(theta)
            name = name+'_beta'+beta_str+'_theta'+theta_str
        name=name+'_sigma'+sigma_str
        cc_params=[beta, theta, sigma]
        run_block_trajopt(friction_erm_params=friction_erm_params, cc_params=cc_params,
                            distance_erm_params=distance_erm_params,
                            uncertainty_option=ermOption, cc_option=ccOption, save_folder=folder, 
                            save_name=name, scale_option=scale_option, tight=tight)
    plot_control_trajectories(folder=folder, name='block_erm', sigmas=friction_sigmas)

def run_block_trajopt(cc_params = [0.5,0.5,0],
                        distance_erm_params = [0.1, 1],
                        friction_erm_params = [0.1, 0.01, 1],
                        uncertainty_option = 1,
                        cc_option = 1,
                        save_folder = None,
                        save_name = None, scale_option=1, tight=False):
    plant, context = create_block_plant()
    options = setup_options()
    trajopt = ChanceConstrainedContactImplicit(plant=plant,
                                            context=context,
                                            num_time_samples=101,
                                            maximum_timestep=0.01,
                                            minimum_timestep=0.01,
                                            chance_param= cc_params,
                                            distance_param = distance_erm_params,
                                            friction_param= friction_erm_params,
                                            optionERM = uncertainty_option,
                                            optionCC = cc_option,
                                            options=options
                                            )
    # Set the boundary constraints
    x0, xf = create_boundary_constraints()
    set_boundary_constraints(trajopt, x0, xf)
    # Require the timesteps be equal
    trajopt.add_equal_time_constraints()
    # Set the costs
    add_control_cost(trajopt, weight=100)
    add_state_cost(trajopt, xf, weight=1)
    # add_final_cost(trajopt)
    # Set the initial condition
    # set_linear_guess(trajopt, x0, xf)
    initialize_from_saved_trajectories(trajopt, folder = "data/IEEE_Access/sliding_block", name="block_trajopt_nominal_tight.pkl")
    # Set the default solver options
    set_default_snopt_options(trajopt, scale_option=scale_option)
    if tight is True:
        set_tight_snopt_options(trajopt)
    #Check the problem for bugs in the constraints
    if not utils.CheckProgram(trajopt.prog):
        quit()
    # Solve the problem
    result = solve_block_trajopt(trajopt, save_folder, save_name)
    soln = trajopt.result_to_dict(result)
    # Plot results
    # plot_block_trajectories(trajopt, result)
    # Save
    save_block_trajectories(soln, folder=save_folder, name=save_name)

def create_block_plant():
    """ Create block plant"""
    plant = Block()
    plant.Finalize()
    # Get the default context
    context = plant.multibody.CreateDefaultContext()
    return plant, context

def setup_nominal_block_trajopt():
    """ Create block plant and contact-implicit trajectory optimization"""
    plant = Block()
    plant.Finalize()
    # Get the default context
    context = plant.multibody.CreateDefaultContext()
    # set up optimization options
    options = setup_options()
    # Create a Contact Implicit Trajectory Optimization
    trajopt = ContactImplicitDirectTranscription(plant=plant,
                                                context=context,
                                                options=options,
                                                num_time_samples=101,
                                                maximum_timestep=0.01,
                                                minimum_timestep=0.01)
    return trajopt

def setup_options():
    """Create options object for contact-implicit trajectory optimization"""
    options = OptimizationOptions()
    # options.ncc_implementation = NCCImplementation.LINEAR_EQUALITY
    options.ncc_implementation = NCCImplementation.NONLINEAR
    options.slacktype = NCCSlackType.CONSTANT_SLACK
    return options

def create_boundary_constraints():
    x0 = np.array([0., 0.5, 0., 0.])
    xf = np.array([5., 0.5, 0., 0.])
    return x0, xf

def set_boundary_constraints(trajopt, x0, xf):
    trajopt.add_state_constraint(knotpoint=0, value=x0)    
    trajopt.add_state_constraint(knotpoint=trajopt.num_time_samples-1, value=xf)

def set_linear_guess(trajopt, x0, xf):
    """
    Set a linear guess for the initial state trajectory
    
    Also sets zero-initialized trajectories for other variables
    """
    # Set the initial trajectory guess
    u_init = np.zeros(trajopt.u.shape)
    x_init = np.zeros(trajopt.x.shape)
    for n in range(0, x_init.shape[0]):
        x_init[n,:] = np.linspace(start=x0[n], stop=xf[n], num=101)
    l_init = np.zeros(trajopt.l.shape)
    # Set the guess in the trajopt
    trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init)

def add_control_cost(trajopt, weight = 1):
    """Add a quadratic cost on the control effort"""
    nU = trajopt.u.shape[0]
    R = weight*np.eye(nU)
    b = np.zeros((nU,))
    trajopt.add_quadratic_running_cost(R, b, [trajopt.u], name="ControlCost")

def add_state_cost(trajopt, b, weight = 1):
    """Add a quadratic cost of the state deviation"""
    nX = trajopt.x.shape[0]
    Q = weight*np.eye(nX)
    trajopt.add_quadratic_running_cost(Q, b, [trajopt.x], name="StateCost")

def add_final_cost(trajopt):
    """Add a final cost on the total time for the motion"""
    cost = lambda h: np.sum(h)
    trajopt.add_final_cost(cost, vars=[trajopt.h], name="TotalTime")

def set_default_snopt_options(trajopt, scale_option = 2):
    """ Set SNOPT solver options """
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 1e6)
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-6)
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-6)
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", scale_option)

def set_tight_snopt_options(trajopt):
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-8)
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-8)

def initialize_from_previous(trajopt, soln):
    trajopt.set_initial_guess(xtraj=soln['state'], utraj=soln['control'], ltraj=soln['force'])

def solve_block_trajopt(trajopt, folder, name):
    solver = SnoptSolver()
    prog = trajopt.get_program()
    # Solve the problem
    print("Solving trajectory optimization")
    start = timeit.default_timer()
    result = solver.Solve(prog)
    stop = timeit.default_timer()
    time = stop - start
    print(f"Elapsed time: {time}")
    folder = folder+'/reports'
    filename = folder +'/' +name+'_report.txt'
    utils.printProgramReport(result, prog, time, filename=filename)
    return result

def plot_block_trajectories(trajopt, result):
    xtraj, utraj, ftraj, _ = trajopt.reconstruct_all_trajectories(result)
    trajopt.plant_f.plot_trajectories(xtraj, utraj, ftraj)

def save_block_trajectories(soln = None, folder=None, name=None):
    file =folder+'/'+name+'.pkl'
    utils.save(file, soln)

def initialize_from_saved_trajectories(trajopt, folder = None, name=None):
    # file = folder + '/' + name
    # soln = utils.load(file)
    # trajopt.set_initial_guess(xtraj=soln['state'], utraj=soln['control'], ltraj=soln['force'])
    x_init = np.loadtxt('data/slidingblock/warm_start/x.txt')
    u_init = np.loadtxt('data/slidingblock/warm_start/u.txt')
    u_init = u_init.reshape(trajopt.u.shape)
    l_init = np.loadtxt('data/slidingblock/warm_start/l.txt')
    trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init)

def ERM_CC_comparison():
    run_block_trajopt_ERM_CC(folder=f"data/IEEE_Access/sliding_block/ERM_v_ERMCC",
        beta=0.65, theta=0.65, scale_option=2, tight=False, ermOption=3, ccOption=3)
    run_block_trajopt_ERM_CC(folder=f"data/IEEE_Access/sliding_block/ERM_v_ERMCC",
        beta=0.65, theta=0.65, scale_option=2, tight=False, ermOption=3, ccOption=1)

def beta_theta():
    betas = np.array([0.51, 0.60, 0.7, 0.8, 0.9])
    thetas = np.array([0.51, 0.60, 0.7, 0.8, 0.9])
    for beta in betas:
        for theta in thetas:
            run_block_trajopt_ERM_CC(folder=f"data/IEEE_Access/sliding_block/beta_theta", 
                beta=beta, theta=theta, tight=False, scale_option=2, ermOption=3, ccOption=3)

def generate_config():
    sigmas = [0.01, 0.05, 0.1, 0.3, 1]
    betas = [0.51, 0.60, 0.7, 0.8, 0.9]
    thetas = [0.51, 0.60, 0.7, 0.8, 0.9]
    configs = []
    for sigma in sigmas:
        for beta in betas:
            for theta in thetas:
                config = [sigma, beta, theta]
                configs.append(config)
    return configs

def run_erm_parallel():
    sigmas = [0.01, 0.05, 0.1, 0.3, 1]
    with concurrent.futures.ProcessPoolExecutor(4) as executor:
        successes = executor.map(run_opt_erm, sigmas)

def run_erm_cc_parallel():
    configs = generate_config()
    with concurrent.futures.ProcessPoolExecutor(4) as executor:
        successes = executor.map(run_opt_erm_cc, configs)

if __name__ == "__main__":
    # ERM_CC_comparison()
    # beta_theta_solvetimes()
    # run_block_trajopt_ERM_CC(folder=f"data/IEEE_Access/sliding_block/ERM",
    #     beta=0.65, theta=0.65, scale_option=1, tight=False, ermOption=3, ccOption=1)
    # beta_theta()
    run_erm_parallel()
    # run_erm_cc_parallel()
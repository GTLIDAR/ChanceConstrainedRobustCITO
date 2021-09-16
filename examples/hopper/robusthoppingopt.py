"""
Robust hopping opt:
Runs contact robust trajectory optimization for the single-legged, footed hopper

Luke Drnach
September 1, 2021
"""
import numpy as np
import os, timeit, concurrent.futures, errno
import utilities as utils
from matplotlib import pyplot as plt
from examples.hopper.hoppingopt import create_hopper, boundary_conditions
from trajopt.robustContactImplicit import ChanceConstrainedContactImplicit
from trajopt.constraints import NCCImplementation
from trajopt.contactimplicit import OptimizationOptions
from pydrake.all import SnoptSolver

class RobustOptimizationOptions():
    def __init__(self):
        self.theta = 0.5
        self.beta = 0.5
        self.sigma = 0.001
        self.erm_multiplier = 10**6
        self.erm_option = 2
        self.cc_option = 1
        self.ncc_mode = OptimizationOptions()
        self.solveroptions = {"Iterations limit": 1000000,
                            'Major iterations limit': 5000,
                            "Major feasibility tolerance": 1e-6,
                            "Major optimality tolerance": 1e-6,
                            "Scale option": 1,
                            "Elastic weight": 10**5
        }
        self.savedir = None
        self.warmstart = None
        self.distance_scale = 1.

    def parse(self):
        return self.chance_params, self.erm_params, self.cc_option, self.erm_option, self.ncc_mode

    def useLinearSlack(self):
        self.ncc_mode.ncc_implementation = NCCImplementation.LINEAR_EQUALITY

    def useNonlinearSlack(self):
        self.ncc_mode.ncc_implementation = NCCImplementation.NONLINEAR

    def useERMOnly(self):
        self.cc_option = 1

    def useDistanceChanceConstraints(self):
        self.cc_option = 2

    def tostring(self):
        if self.ncc_mode.ncc_implementation == NCCImplementation.LINEAR_EQUALITY:
            base = 'linear_NCC'
        else:
            base = 'nonlinear_NCC'
        if self.cc_option == 1:
            return f"{base}_sigma_{self.sigma:.0e}_nochance"
        else:
            return f"{base}_sigma_{self.sigma:.0e}_theta_{self.theta:.0e}_beta_{self.beta:.0e}"

    def to_dict(self):
        return {'sigma': self.sigma,
                'theta': self.theta,
                'beta': self.beta,
                'erm_multiplier': self.erm_multiplier,
                'erm_mode': self.erm_option,
                'chance_mode': self.cc_option,
                'ncc_mode': self.ncc_mode.ncc_implementation}

    @property
    def chance_params(self):
        return [self.beta, self.theta, self.sigma]

    @chance_params.setter
    def chance_params(self, vals):
        self.beta, self.theta, self.sigma = vals[0], vals[1], vals[2]

    @property
    def erm_params(self):
        return [self.sigma, self.erm_multiplier]

    @erm_params.setter
    def erm_params(self, vals):
        self.sigma, self.erm_multiplier = vals[0], vals[1]

def copy_config_to_data(config, data):
    data['sigma'] = config.sigma
    data['theta'] = config.theta
    data['beta'] = config.beta
    data['chanceconstraints'] = (config.cc_option == 2)
    data['erm'] = (config.erm_option == 2)
    return data

def get_reference_solution_linear():
    file = os.path.join("examples","hopper","reference_linear","strict_linear_equality","trajoptresults.pkl")
    data = utils.load(file)
    return data['state'], data['control'], data['force'], data['jointlimit'], data['slacks']

def get_reference_solution():
    file = os.path.join("examples","hopper","reference_linear","strict_nocost","trajoptresults.pkl")
    data = utils.load(file)       
    return data['state'], data['control'], data['force'], data['jointlimit']

def get_warmstart_from_file(filename):
    if os.path.isfile(filename):
        data = utils.load(filename)
        return data['state'], data['control'], data['force'], data['jointlimit']
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

def create_robust_optimization(robustconfig):
    """Create a robust trajectory optimization for the footed hopper"""
    # parse the robust config
    chance_params, erm_params, chance_mode, erm_mode, ncc_mode = robustconfig.parse()
    
    # Create the hopper
    hopper = create_hopper()
    hopper.terrain.friction = 1.
    x0, xf = boundary_conditions(hopper)
    # Create the trajectory optimization
    max_time = 3
    min_time = 3
    N = 101
    trajopt = ChanceConstrainedContactImplicit(hopper, 
                    hopper.multibody.CreateDefaultContext(),
                    num_time_samples=N,
                    minimum_timestep = min_time/(N-1),
                    maximum_timestep = max_time/(N-1),
                    chance_param = chance_params,
                    distance_param = erm_params,
                    optionCC = chance_mode,
                    optionERM = erm_mode,
                    options = ncc_mode)
    # Add the state constraints
    trajopt.add_state_constraint(knotpoint=0, value=x0)
    trajopt.add_state_constraint(knotpoint=N-1, value=xf)
    # Require equal timesteps
    trajopt.add_equal_time_constraints()
    # Add the running costs
    R = 0.01*np.eye(3)
    Q = np.diag([1, 10, 10, 100, 100, 1, 1, 1, 1, 1])
    R = R/2
    Q = Q/2
    trajopt.add_quadratic_running_cost(R, np.zeros((3,)), vars=[trajopt.u], name='ControlCost')
    trajopt.add_quadratic_running_cost(Q, xf, vars=[trajopt.x], name='StateCost')
    # Get the initial guess and set it
    if robustconfig.warmstart is not None:
        xtraj, utraj, ltraj, jltraj = get_warmstart_from_file(robustconfig.warmstart)
        trajopt.set_initial_guess(xtraj=xtraj, utraj=utraj, ltraj=ltraj, jltraj=jltraj)
    elif ncc_mode.ncc_implementation == NCCImplementation.LINEAR_EQUALITY:
        xtraj, utraj, ltraj, jltraj, straj = get_reference_solution_linear()
        trajopt.set_initial_guess(xtraj=xtraj, utraj=utraj, ltraj=ltraj, jltraj=jltraj, straj=straj)
    else:
        xtraj, utraj, ltraj, jltraj = get_reference_solution()
        trajopt.set_initial_guess(xtraj=xtraj, utraj=utraj, ltraj=ltraj, jltraj=jltraj)
    #Turn on cost display
    trajopt.enable_cost_display('figure', title=robustconfig.tostring())
    # Set the reaction force scaling
    trajopt.force_scale = 1
    # Set the distance scaling
    trajopt.distance_scale = robustconfig.distance_scale
    return trajopt

def solve_robust_optimization(trajopt, config, savedir=None):

    """Solve the trajectory optimization problem and return the solution dictionary"""
    # Create the program
    solver = SnoptSolver()
    solverid = solver.solver_id()
    prog = trajopt.get_program()
    # Set SNOPT options
    for key in config.solveroptions.keys():
        prog.SetSolverOption(solverid, key, config.solveroptions[key])
    # prog.SetSolverOption(solverid, "Iterations limit", 100000)
    # prog.SetSolverOption(solverid, 'Major iterations limit',5000)
    # prog.SetSolverOption(solverid, "Major feasibility tolerance", 1e-6)
    # prog.SetSolverOption(solverid, "Major optimality tolerance", 1e-6)
    # prog.SetSolverOption(solverid, "Scale option",1)
    # prog.SetSolverOption(solverid, "Elastic weight", 10**5)

    if not utils.CheckProgram(prog):
        quit()
    # Time and solve the optimization
    print(f"Process {os.getpid()}: Solving trajectory optimization")
    start = timeit.default_timer()
    result = solver.Solve(prog)
    stop = timeit.default_timer()
    print(f"Process {os.getpid()}: Elapsed time: {stop-start}")
    # Print the details of the solution
    print(f"Process {os.getpid()}: Optimization successful? {result.is_success()}")
    print(f"Process {os.getpid()}: Solved with {result.get_solver_id().name()}")
    print(f"Process {os.getpid()}: Optimal cost = {result.get_optimal_cost()}")
    # Convert results to dictionary
    solndict = trajopt.result_to_dict(result)
    # Append info to solndict
    solndict['elapsed'] = stop - start
    # Copy config to solndict
    solndict = copy_config_to_data(config, solndict)
    # Make figures and save
    print(f"savedir is {savedir}")
    if savedir is not None:
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        # Save the report
        utils.printProgramReport(result, prog, terminal=False, filename=os.path.join(savedir,'report.txt'), verbose=True)
        # Save the cost figure
        trajopt.printer.save_and_close(os.path.join(savedir,'CostsAndConstraints.png'))
        # Save the data
        utils.save(os.path.join(savedir, 'trajoptresults.pkl'), solndict)
        utils.save(os.path.join(savedir, 'config.pkl'), config)
        # Plot and save trajectories
        xtraj, utraj, ftraj, jltraj, _ = trajopt.reconstruct_all_trajectories(result)
        figs, _ = trajopt.plant_f.plot_trajectories(xtraj, utraj, ftraj, jltraj, show=False, savename=os.path.join(savedir, 'opt.png'))
        for fig in figs:
            plt.close(fig)
        return solndict['success']
    else:
        return solndict

def run_optimization(robustconfig):
    #basedir = os.path.join('examples','hopper','robust_erm_hotfix_1e6_linear')
    basedir = robustconfig.savedir
    trajopt = create_robust_optimization(robustconfig)
    savedir = os.path.join(basedir, robustconfig.tostring())
    status = solve_robust_optimization(trajopt, robustconfig, savedir)
    return status

def make_robust_erm_options():

    #sigmas = [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7]
    sigmas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    configlist = []
    for sigma in sigmas:
        configlist.append(RobustOptimizationOptions())
        configlist[-1].sigma = sigma
        configlist[-1].useERMOnly()
        configlist[-1].useLinearSlack()

    return configlist 

def make_robust_cc_options():
    #sigmas = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7]
    sigmas = [0.1, 0.2, 0.25, 0.4, 0.6, 0.7]
    #sigmas = [0.07, 0.5, 0.7]
    thetas = [0.51, 0.6, 0.7, 0.8, 0.9]
    #betas = [0.5, 0.6, 0.7, 0.8, 0.9]
    configlist = []
    for sigma in sigmas:
        for theta in thetas:
            configlist.append(RobustOptimizationOptions())
            configlist[-1].sigma = sigma
            configlist[-1].theta = theta
            configlist[-1].beta = 0.5
            configlist[-1].useLinearSlack()
            configlist[-1].useDistanceChanceConstraints()
    return configlist

def main_cc(basedir):
    configs = make_robust_cc_options()
    for n in range(len(configs)):
        #configs[n].solveroptions['Scale option'] = 2
        configs[n].savedir = basedir
    with concurrent.futures.ProcessPoolExecutor(10) as executor:
        successes = executor.map(run_optimization, configs)
    successes = [success for success in successes]
    print(f"{sum(successes)} of {len(successes)} solved succcessfully")

def main_erm(basedir):
    configs = make_robust_erm_options()
    for n in range(len(configs)):
        #configs[n].solveroptions['Scale option'] = 2
        configs[n].savedir = basedir
    with concurrent.futures.ProcessPoolExecutor(10) as executor:
        successes = executor.map(run_optimization, configs)
    successes = [success for success in successes]
    print(f"{sum(successes)} of {len(successes)} solved successfully")

def main_cc_tight(basedir):
    configs = make_robust_cc_options()
    for n in range(len(configs)):
        configs[n].solveroptions['Major feasibility tolerance'] = 1e-8
        configs[n].solveroptions['Major optimality tolerance'] = 1e-8
        configs[n].savedir = basedir
    with concurrent.futures.ProcessPoolExecutor(10) as executor:
        successes = executor.map(run_optimization, configs)
    successes = [success for success in successes]
    print(f"{sum(successes)} of {len(successes)} solved succcessfully")

def main_erm_tight(basedir):
    configs = make_robust_erm_options()
    for n in range(len(configs)):
        configs[n].solveroptions['Major feasibility tolerance'] = 1e-8
        configs[n].solveroptions['Major optimality tolerance'] = 1e-8
        configs[n].savedir = basedir
    with concurrent.futures.ProcessPoolExecutor(10) as executor:
        successes = executor.map(run_optimization, configs)
    successes = [success for success in successes]
    print(f"{sum(successes)} of {len(successes)} solved successfully")

def main_erm_hotfix():
    config = RobustOptimizationOptions()
    config.sigma = 1e-01
    config.useLinearSlack()
    config.useERMOnly()
    config.erm_multiplier = 10**4
    success = run_optimization(config)
    print(f'success: {success}')

def main_erm_lowmult(basedir):
    configs = make_robust_erm_options()
    for n in range(len(configs)):
        configs[n].erm_multiplier = 10**4
        configs[n].savedir = basedir
    with concurrent.futures.ProcessPoolExecutor(10) as executor:
        successes = executor.map(run_optimization, configs)
    successes = [success for success in successes]
    print(f"{sum(successes)} of {len(successes)} solved successfully")

def make_lowmult_cc_options():
    #sigmas = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7]
    sigmas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    #sigmas = [0.07, 0.5, 0.7]
    thetas = [0.51, 0.6, 0.7, 0.8, 0.9]
    #betas = [0.5, 0.6, 0.7, 0.8, 0.9]
    configlist = []
    for sigma in sigmas:
        for theta in thetas:
            configlist.append(RobustOptimizationOptions())
            configlist[-1].sigma = sigma
            configlist[-1].theta = theta
            configlist[-1].beta = 0.5
            configlist[-1].useLinearSlack()
            configlist[-1].useDistanceChanceConstraints() 
    return configlist

def main_cc_lowmult(basedir):
    configs = make_lowmult_cc_options()
    runConfigs = []
    for n in range(len(configs)):
        configs[n].erm_multiplier = 10**4
        configs[n].savedir = basedir
        # Check if the configuration has already been solved
        savepath = os.path.join(basedir, 'success','cc',configs[n].tostring())
        failpath = os.path.join(basedir, 'fail', 'cc', configs[n].tostring())
        if not os.path.isdir(savepath) or not os.path.isdir(failpath):
            runConfigs.append(configs[n])
    with concurrent.futures.ProcessPoolExecutor(10) as executor:
        successes = executor.map(run_optimization, runConfigs)
    successes = [success for success in successes]
    print(f"{sum(successes)} of {len(successes)} solved successfully")


def make_erm_nonlinear_options(basedir=None):
    sigmas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    configlist = []
    for sigma in sigmas:
        configlist.append(RobustOptimizationOptions())
        configlist[-1].sigma = sigma
        configlist[-1].useERMOnly()
        configlist[-1].useNonlinearSlack()
        configlist[-1].savedir = basedir
        configlist[-1].erm_multiplier = 10**5
        configlist[-1].solveroptions['Scale option'] = 2
        configlist[-1].warmstart = os.path.join('examples','hopper','reference_highfriction','strict_scale2','trajoptresults.pkl')
    return configlist

def make_erm_warmstarted_from_chance(basedir=None):
    sigmas = [0.01, 0.05, 0.1, 0.3, 0.4]
    configlist = []
    for sigma in sigmas:
        configlist.append(RobustOptimizationOptions())
        configlist[-1].sigma = sigma
        configlist[-1].useERMOnly()
        configlist[-1].useNonlinearSlack()
        configlist[-1].savedir = basedir
        configlist[-1].erm_multiplier = 10**5
        configlist[-1].warmstart = os.path.join("examples","hopper","robust_nonlinear","cc_erm_1e5_mod_ermstart","success",f"nonlinear_NCC_sigma_{sigma:.0e}_theta_9e-01_beta_5e-01","trajoptresults.pkl")
    return configlist

def main_erm_warmstarted(basedir):
    configs = make_erm_warmstarted_from_chance(basedir)
    run_configs_parallel(configs)

def main_erm_nonlinear(basedir):
    configs = make_erm_nonlinear_options(basedir)
    run_configs_parallel(configs)

def run_configs_parallel(configs):
    with concurrent.futures.ProcessPoolExecutor(10) as executor:
        successes = executor.map(run_optimization, configs)
    successes = [success for success in successes]
    print(f"{sum(successes)} of {len(successes)} solved successfully")

def make_cc_nonlinear_options(basedir):
    sigmas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    thetas = [0.51, 0.6, 0.7, 0.8, 0.9]
    configlist = []
    for sigma in sigmas:
        for theta in thetas:
            configlist.append(RobustOptimizationOptions())
            configlist[-1].sigma = sigma
            configlist[-1].theta = theta
            configlist[-1].beta = 0.5
            configlist[-1].useDistanceChanceConstraints()
            configlist[-1].useNonlinearSlack()
            configlist[-1].savedir = basedir
            configlist[-1].erm_multiplier = 10**5
            configlist[-1].warmstart = os.path.join('examples','hopper','reference_highfriction','strict','trajoptresults.pkl')
    return configlist


def main_cc_nonlinear(basedir):
    configs = make_cc_nonlinear_options(basedir)
    run_configs_parallel(configs)


def make_erm_lengthscaled(basedir):
    sigmas = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0]
    configlist = []
    for sigma in sigmas:
        configlist.append(RobustOptimizationOptions())
        configlist[-1].sigma = sigma
        configlist[-1].useERMOnly()
        configlist[-1].useNonlinearSlack()
        configlist[-1].savedir = basedir
        configlist[-1].erm_multiplier = 10**3
        configlist[-1].distance_scale = 10
        configlist[-1].warmstart = os.path.join('examples','hopper','reference_linear','strict_nocost','trajoptresults.pkl')
    return configlist

def main_erm_lengthscaled(basedir):
    configs = make_erm_lengthscaled(basedir)
    run_configs_parallel(configs)

def make_cc_lengthscaled(basedir):
    sigmas = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0]
    thetas = [0.51, 0.6, 0.7, 0.8, 0.9]
    configlist = []
    for sigma in sigmas:
        for theta in thetas:
            configlist.append(RobustOptimizationOptions())
            configlist[-1].sigma = sigma
            configlist[-1].theta = theta
            configlist[-1].beta = 0.5
            configlist[-1].useERMOnly()
            configlist[-1].useNonlinearSlack()
            configlist[-1].savedir = basedir
            configlist[-1].erm_multiplier = 10**3
            configlist[-1].distance_scale = 10
            configlist[-1].warmstart = os.path.join('examples','hopper','reference_linear','strict_nocost','trajoptresults.pkl')
    return configlist

def main_cc_lengthscaled(basedir):
    configs = make_cc_lengthscaled(basedir)
    run_configs_parallel(configs)

def make_cc_lengthscaled_warmstarted(basedir):
    sigmas = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0]
    thetas = [0.51, 0.6, 0.7, 0.8, 0.9]
    configlist = []
    for sigma in sigmas:
        for theta in thetas:
            configlist.append(RobustOptimizationOptions())
            configlist[-1].sigma = sigma
            configlist[-1].theta = theta
            configlist[-1].beta = 0.5
            configlist[-1].useERMOnly()
            configlist[-1].useNonlinearSlack()
            configlist[-1].savedir = basedir
            configlist[-1].erm_multiplier = 10**3
            configlist[-1].distance_scale = 10
            configlist[-1].warmstart = os.path.join('examples','hopper','robust_nonlinear','decimeters_1e3',"erm", f"nonlinear_NCC_sigma_{sigma:.0e}_nochance", 'trajoptresults.pkl')
    return configlist

def main_cc_lengthscaled_warmstarted(basedir):
    configs = make_cc_lengthscaled_warmstarted(basedir)
    run_configs_parallel(configs)

if __name__ == "__main__":
    # main_erm(basedir = os.path.join('examples','hopper','robust_erm_hotfix_1e6_linear_take2','erm'))
    # main_erm_tight(basedir = os.path.join('examples','hopper','robust_erm_linear_hotfix_tol1e-8_scale1_erm'))
    # main_cc_tight(basedir = os.path.join('examples','hopper','robust_linear_hotfix_tol1e-8_scale1','erm_cc'))
    #lowdir = os.path.join('examples','hopper','robust_hotfix_linear_1e4')
    #main_erm_lowmult(os.path.join(lowdir, 'erm'))
    #main_cc_lowmult(os.path.join(lowdir,'cc'))
    #main_cc(basedir = os.path.join('examples','hopper','robust_erm_hotfix_1e6_linear_take2','erm_cc'))
    #main_erm_nonlinear(basedir = os.path.join('examples','hopper','robust_nonlinear','erm_1e5_scale2'))
    #main_cc_nonlinear(basedir = os.path.join('examples','hopper','robust_nonlinear','cc_erm_1e5_mod'))
    #main_erm_warmstarted(basedir=os.path.join("examples","hopper","robust_nonlinear","erm_1e5_warmstarted_from_chance"))
    main_erm_lengthscaled(basedir=os.path.join("examples","hopper","robust_nonlinear","decimeters_1e3", "erm"))
    main_cc_lengthscaled(basedir=os.path.join("examples","hopper","robust_nonlinear","decimeters_1e3","erm_cc"))
    main_cc_lengthscaled_warmstarted(basedir=os.path.join("examples","hopper","robust_nonlinear","decimeters_1e3","erm_cc_warmstarted"))
    #main_cc_nonlinear(basedir = os.path.join('example','hopper','robust_highfriction','cc_erm_1e5'))
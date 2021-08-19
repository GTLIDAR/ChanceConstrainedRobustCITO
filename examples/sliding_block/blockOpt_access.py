"""
Contact Implicit Trajectory Optimization for a sliding block. 
This script is used mainly to test the implementation of Contact Implicit Trajectory Optimization in contactimplicit.py

The goal is to move a 1kg block 5m in 1s. The timesteps are fixed, and the objective is to minimize the control cost and the state deviation from the final position. Boundary constraints are added to the problem to ensure the block starts and stops at rest and at the desired positions. In this example, the timesteps are fixed and equal. 

Luke Drnach
October 15, 2020
"""
# Imports
import timeit
import numpy as np
import matplotlib.pyplot as plt
from trajopt.contactimplicit import ContactImplicitDirectTranscription, OptimizationOptions
from trajopt.constraints import NCCImplementation, NCCSlackType
from systems.block.block import Block
from pydrake.solvers.snopt import SnoptSolver
import utilities as utils
#TODO: Check if the ElasticWeight parameter is available in this version of Drake

def run_block_trajopt():
    trajopt = setup_nominal_block_trajopt()
    # Set the boundary constraints
    x0, xf = create_boundary_constraints()
    set_boundary_constraints(trajopt, x0, xf)
    # Require the timesteps be equal
    trajopt.add_equal_time_constraints()
    # Set the costs
    add_control_cost(trajopt)
    add_state_cost(trajopt, xf)
    add_final_cost(trajopt)
    # Set the initial condition
    set_linear_guess(trajopt, x0, xf)
    # Set the default solver options
    set_default_snopt_options(trajopt)
    #Check the problem for bugs in the constraints
    if not utils.CheckProgram(trajopt.prog):
        quit()
    # Solve the problem
    result = solve_block_trajopt(trajopt)
    soln = trajopt.result_to_dict(result)
    # Plot results
    plot_block_trajectories(trajopt, result)
    # Save
    save_block_trajectories(soln, 'block_trajopt.pkl')
    # Tighten snopt options
    # set_tight_snopt_options(trajopt)
    # initialize_from_previous(trajopt, soln)
    # # Run optimization
    # result = solve_block_trajopt(trajopt)
    # soln = trajopt.result_to_dict(result)
    # # Plot results
    # plot_block_trajectories(trajopt, result)
    # # Save
    # save_block_trajectories(soln, 'block_trajopt_tight.pkl')

def setup_nominal_block_trajopt():
    """ Create block plant and contact-implicit trajectory optimization"""
    plant = Block()
    plant.Finalize()
    # Get the default context
    context = plant.multibody.CreateDefaultContext()
    # set up optimization options
    options = OptimizationOptions()
    options.ncc_implementation = NCCImplementation.LINEAR_EQUALITY
    options.slacktype = NCCSlackType.CONSTANT_SLACK
    # Create a Contact Implicit Trajectory Optimization
    trajopt = ContactImplicitDirectTranscription(plant=plant,
                                                context=context,
                                                # options=options,
                                                num_time_samples=101,
                                                maximum_timestep=0.01,
                                                minimum_timestep=0.01,
                                                )
    
    return trajopt

def setup_options():
    """Create options object for contact-implicit trajectory optimization"""
    options = OptimizationOptions()
    options.ncc_implementation = NCCImplementation.LINEAR_EQUALITY
    options.slacktype = NCCSlackType.CONSTANT_SLACK
    return options
# def setup_nominal_block_trajopt():
#     """ Create block plant and robust contact-implicit trajectory optimization""""
#     # plant = Block()
#     # plant.Finalize()

#     pass
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

def add_control_cost(trajopt):
    """Add a quadratic cost on the control effort"""
    nU = trajopt.u.shape[0]
    R = 10 * np.eye(nU)
    b = np.zeros((nU,))
    trajopt.add_quadratic_running_cost(R, b, [trajopt.u], name="ControlCost")

def add_state_cost(trajopt, b):
    """Add a quadratic cost of the state deviation"""
    nX = trajopt.x.shape[0]
    R = np.eye(nX)
    trajopt.add_quadratic_running_cost(R, b, [trajopt.x], name="StateCost")

def add_final_cost(trajopt):
    """Add a final cost on the total time for the motion"""
    cost = lambda h: np.sum(h)
    trajopt.add_final_cost(cost, vars=[trajopt.h], name="TotalTime")

def set_default_snopt_options(trajopt):
    """ Set SNOPT solver options """
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 10000)
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-6)
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-6)
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 2)

def set_tight_snopt_options(trajopt):
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-8)

def initialize_from_previous(trajopt, soln):
    trajopt.set_initial_guess(xtraj=soln['state'], utraj=soln['control'], ltraj=soln['force'])

def solve_block_trajopt(trajopt):
    solver = SnoptSolver()
    prog = trajopt.get_program()
    # Solve the problem
    print("Solving trajectory optimization")
    start = timeit.default_timer()
    result = solver.Solve(prog)
    stop = timeit.default_timer()
    print(f"Elapsed time: {stop-start}")
    utils.printProgramReport(result, prog)
    return result

def plot_block_trajectories(trajopt, result):
    xtraj, utraj, ftraj, _ = trajopt.reconstruct_all_trajectories(result)
    trajopt.plant_f.plot_trajectories(xtraj, utraj, ftraj)

def save_block_trajectories(soln, folder = "data/slidingblock/", name="block_trajopt.pkl"):
    file = folder + name
    utils.save(file, soln)

if __name__ == "__main__":
    run_block_trajopt()
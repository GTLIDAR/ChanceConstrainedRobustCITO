"""
Check the hopper reference trajectory

"""
from examples.hopper.hoppingopt import create_hopper, boundary_conditions
import utilities as utils
import os
import trajopt.contactimplicit as ci
import numpy as np
from pydrake.all import SnoptSolver
import matplotlib.pyplot as plt
from trajopt.constraints import NCCImplementation

def get_reference_solution(trajopt, ncc_mode, file):
    data = utils.load(file)
    if ncc_mode.ncc_implementation != NCCImplementation.LINEAR_EQUALITY:
        # Calculate the required slack variables
        numN = trajopt.numN
        numT = trajopt.numT
        straj = np.zeros((2*numN + numT, data['state'].shape[1]))
        for n in range(data['state'].shape[1]):
            straj[:numN,n] = trajopt._normal_distance(data['state'][:,n])
            straj[numN:numN+numT,n] = trajopt._sliding_velocity(np.concatenate([data['state'][:,n], data['force'][numN+numT:,n]], axis=0))
            straj[numN+numT:, n] = trajopt._friction_cone(np.concatenate([data['state'][:,n], data['force'][:numN+numT,n]], axis=0))
    else:
        straj = None
        
    return data['state'], data['control'], data['force'], data['jointlimit'], straj

def create_hopper_opt(hopper, x0, xf, N=101):
    context = hopper.multibody.CreateDefaultContext()
    # Create the optimization
    max_time = 3
    min_time = 3
    options = ci.OptimizationOptions()
    trajopt = ci.ContactImplicitDirectTranscription(hopper, context, num_time_samples=N, minimum_timestep=min_time/(N-1), maximum_timestep=max_time/(N-1), options=options)
    # Add boundary constraints
    trajopt.add_state_constraint(knotpoint=0, value=x0)
    trajopt.add_state_constraint(knotpoint=trajopt.num_time_samples-1, value=xf)
    trajopt.enable_cost_display('figure')
    # Require equal timesteps
    trajopt.add_equal_time_constraints()
    # Set the force scaling
    trajopt.force_scaling = 1
    # Add in running cost
    R = 0.01*np.eye(3)
    Q = np.diag([1, 10, 10, 100, 100, 1, 1, 1, 1, 1])
    R = R/2
    Q = Q/2
    trajopt.add_quadratic_running_cost(R, np.zeros((3,)), vars=[trajopt.u], name='ControlCost')
    trajopt.add_quadratic_running_cost(Q, xf, vars=[trajopt.x], name='StateCost')

    return trajopt

def create_hopper_opt_linear(hopper, x0, xf, N=101):
    context = hopper.multibody.CreateDefaultContext()
    # Create the optimization
    max_time = 3
    min_time = 3
    options = ci.OptimizationOptions()
    options.ncc_implementation = NCCImplementation.LINEAR_EQUALITY
    trajopt = ci.ContactImplicitDirectTranscription(hopper, context, num_time_samples=N, minimum_timestep=min_time/(N-1), maximum_timestep=max_time/(N-1), options=options)
    # Add boundary constraints
    trajopt.add_state_constraint(knotpoint=0, value=x0)
    trajopt.add_state_constraint(knotpoint=trajopt.num_time_samples-1, value=xf)
    trajopt.enable_cost_display('figure')
    # Require equal timesteps
    trajopt.add_equal_time_constraints()
    # Set the force scaling
    trajopt.force_scaling = 1
    # Add in running cost
    R = 0.01*np.eye(3)
    Q = np.diag([1, 10, 10, 100, 100, 1, 1, 1, 1, 1])
    R = R/2
    Q = Q/2
    trajopt.add_quadratic_running_cost(R, np.zeros((3,)), vars=[trajopt.u], name='ControlCost')
    trajopt.add_quadratic_running_cost(Q, xf, vars=[trajopt.x], name='StateCost')

    return trajopt

def make_figures_and_save(trajopt, result, savedir):
    # Save the outputs
    report = utils.printProgramReport(result, trajopt.prog, terminal=False, filename=os.path.join(savedir, 'report.txt'), verbose=True)
    utils.save(os.path.join(savedir, 'trajoptresults.pkl'), trajopt.result_to_dict(result))
    # Save the cost figure
    trajopt.printer.save_and_close(os.path.join(savedir, 'CostsAndConstraints.png'))
    # Plot and save the trajectories
    xtraj, utraj, ftraj, jltraj, _ = trajopt.reconstruct_all_trajectories(result)
    figs, _ = trajopt.plant_f.plot_trajectories(xtraj, utraj, ftraj, jltraj, show=False, savename=os.path.join(savedir, 'opt.png'))
    for fig in figs:
        plt.close(fig)

def main(filename, outdir=None):
    # Create the optimization
    hopper = create_hopper()
    x0, xf = boundary_conditions(hopper)
    trajopt = create_hopper_opt(hopper, x0, xf)
    # Load the warmstart
    data = utils.load(filename)
    # Initialize the trajectory optimization
    trajopt.set_initial_guess(xtraj=data['state'], utraj=data['control'], ltraj=data['force'], jltraj=data['jointlimit'])
    # Solve the resulting problem
    solver = SnoptSolver()
    solverid = solver.solver_id()
    trajopt.prog.SetSolverOption(solverid, "Iterations limit", 100000)
    trajopt.prog.SetSolverOption(solverid, "Major iterations limit", 100)
    trajopt.prog.SetSolverOption(solverid, "Minor iterations limit", 1000)
    trajopt.prog.SetSolverOption(solverid, "Superbasics limit", 1500)
    trajopt.prog.SetSolverOption(solverid, "Scale option", 1)
    trajopt.prog.SetSolverOption(solverid, "Elastic Weight", 10**5)
    #Solve it
    result = solver.Solve(trajopt.prog)
    if outdir is not None:
        make_figures_and_save(trajopt, result, outdir)
    return result.is_success()

def main_highfriction(filename, outdir=None):
    # Create the optimization
    hopper = create_hopper()
    hopper.terrain.friction = 1.
    x0, xf = boundary_conditions(hopper)
    trajopt = create_hopper_opt(hopper, x0, xf)
    # Load the warmstart
    data = utils.load(filename)
    # Initialize the trajectory optimization
    trajopt.set_initial_guess(xtraj=data['state'], utraj=data['control'], ltraj=data['force'], jltraj=data['jointlimit'])
    # Solve the resulting problem
    solver = SnoptSolver()
    solverid = solver.solver_id()
    trajopt.prog.SetSolverOption(solverid, "Iterations limit", 100000)
    trajopt.prog.SetSolverOption(solverid, "Major iterations limit", 100)
    trajopt.prog.SetSolverOption(solverid, "Minor iterations limit", 1000)
    trajopt.prog.SetSolverOption(solverid, "Superbasics limit", 1500)
    trajopt.prog.SetSolverOption(solverid, "Scale option", 1)
    trajopt.prog.SetSolverOption(solverid, "Elastic Weight", 10**5)
    #Solve it
    result = solver.Solve(trajopt.prog)
    if outdir is not None:
        make_figures_and_save(trajopt, result, outdir)
    return result.is_success()

def main_linear(file, outdir):
    # Create the optimization
    hopper = create_hopper()
    x0, xf = boundary_conditions(hopper)
    trajopt = create_hopper_opt_linear(hopper, x0, xf)
    # Initialize the trajectory optimization
    xinit, uinit, linit, jlinit, sinit = get_reference_solution(trajopt, trajopt.options, file)
    trajopt.set_initial_guess(xtraj=xinit, utraj=uinit, ltraj=linit, jltraj=jlinit, straj=sinit)
    # Solve the resulting problem
    solver = SnoptSolver()
    solverid = solver.solver_id()
    trajopt.prog.SetSolverOption(solverid, "Iterations limit", 100000)
    trajopt.prog.SetSolverOption(solverid, "Major iterations limit", 100)
    trajopt.prog.SetSolverOption(solverid, "Minor iterations limit", 1000)
    trajopt.prog.SetSolverOption(solverid, "Superbasics limit", 1500)
    trajopt.prog.SetSolverOption(solverid, "Scale option", 2)
    trajopt.prog.SetSolverOption(solverid, "Elastic Weight", 10**5)
    #Solve it
    result = solver.Solve(trajopt.prog)
    if outdir is not None:
        make_figures_and_save(trajopt, result, outdir)
    return result.is_success()


if __name__ == "__main__": 
    file = os.path.join("examples","hopper","reference_highfriction","weight_10000","trajoptresults.pkl")
    outdir = os.path.join("examples","hopper","reference_highfriction","strict")
    success = main_highfriction(file, outdir)
    print(f"Check successful? {success}")
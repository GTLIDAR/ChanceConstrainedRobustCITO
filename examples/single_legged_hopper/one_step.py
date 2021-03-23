# Imports
import timeit
import numpy as np
import matplotlib.pyplot as plt
# from trajopt.contactimplicit import ContactImplicitDirectTranscription
from trajopt.robustContactImplicit import ChanceConstrainedContactImplicit
from systems.timestepping import TimeSteppingMultibodyPlant
from pydrake.solvers.snopt import SnoptSolver
import utilities as utils
from scipy.special import erfinv
import pickle
import os
from tempfile import TemporaryFile
from plot_hopper import plot_CC
# Create the block model with the default flat terrain
_file = "systems/urdf/single_legged_hopper.urdf"
plant = TimeSteppingMultibodyPlant(file= _file)
plant.Finalize()
num_step = 101
step_size = 0.01
# Get the default context
context = plant.multibody.CreateDefaultContext()
# set chance constraints parameters
beta, theta = 0.6, 0.6
sigmas = [0.2, 0.3, 0.4]
# sigmas = np.array([0.28])
times = []
# set distance ERM parameters
distance_multiplier = 1e5
# set uncertainty option
uncertainty_option = 2
# set chance constraint option
cc_option = 1
# Add initial and final state constraints
angle = 0.5 
height = np.cos(0.5) * 2
# Add initial and final state
x0 = np.array([0, height, -angle, 2*angle, -angle, 0, 0, 0, 0, 0])
xf = np.array([2, height, -angle, 2*angle, -angle, 0, 0, 0, 0, 0])

# iterations 
iteration = len(sigmas)

# initial trajectory
x_init = np.loadtxt('data/single_legged_hopper/nominal/x1.txt')
u_init = np.loadtxt('data/single_legged_hopper/nominal/u1.txt')
l_init = np.loadtxt('data/single_legged_hopper/nominal/l1.txt')
t_init = np.loadtxt('data/single_legged_hopper/nominal/t1.txt')
# loop through different variance values
for i in range (iteration):
    sigma = sigmas[i]
    chance_params = np.array([beta, theta, sigma])
    distance_erm_params = np.array([sigma, distance_multiplier])
    # Create a Contact Implicit Trajectory Optimization
    trajopt = ChanceConstrainedContactImplicit(plant=plant,
                                            context=context,
                                            num_time_samples=num_step,
                                            maximum_timestep=step_size,
                                            minimum_timestep=step_size,
                                            chance_param= chance_params,
                                            distance_param= distance_erm_params,
                                            optionERM = uncertainty_option,
                                            optionCC = cc_option)

    u_init = u_init.reshape(trajopt.u.shape)
    trajopt.add_state_constraint(knotpoint=0, value=x0)    
    trajopt.add_state_constraint(knotpoint=100, value=xf)
    # Set all the timesteps to be equal
    trajopt.add_equal_time_constraints()
    # Add control and state costs
    # Add a running cost weights on the controls
    R=  0.01*np.diag([1,1,1])
    b = np.zeros((3,))
    # Add running cost weight on state
    Q = np.diag([1,10,10,100,100,1,1,10,10,10])
    trajopt.add_quadratic_running_cost(R, b, [trajopt.u], name="ControlCost")
    trajopt.add_quadratic_running_cost(Q, xf, [trajopt.x], name="StateCost")

    u_init = u_init.reshape(trajopt.u.shape)
    # Set the initial trajectory guess, might switch out later for a warm start trajectory

    trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init)
    prog = trajopt.get_program()
    
    # Set the SNOPT solver options
    prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 1e5)
    prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-6)
    prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-6)
    prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 1)
    solver = SnoptSolver()
    # Solve the problem
    print("Solving trajectory optimization ", i + 1)
    start = timeit.default_timer()
    result = solver.Solve(prog)
    stop = timeit.default_timer()
    print(f"Elapsed time: {stop-start}")
    times.append(stop - start)
    # Print the details of the solution
    print(f"Optimization successful? {result.is_success()}")
    print(f"Solved with {result.get_solver_id().name()}")
    print(f"Optimal cost = {result.get_optimal_cost()}")
    # Get the exit code from SNOPT
    print(f"SNOPT Exit Status {result.get_solver_details().info}: {utils.SNOPT_DECODER[result.get_solver_details().info]}")
    # Unpack and plot the trajectories
    x = trajopt.reconstruct_state_trajectory(result)
    u = trajopt.reconstruct_input_trajectory(result)
    l = trajopt.reconstruct_reaction_force_trajectory(result)
    t = trajopt.get_solution_times(result)
    # save trajectory
    np.savetxt('data/single_legged_hopper/erm/x_{f}.txt'.format(f = sigma), x, fmt = '%1.3f')
    np.savetxt('data/single_legged_hopper/erm/u_{f}.txt'.format(f = sigma), u, fmt = '%1.3f')
    np.savetxt('data/single_legged_hopper/erm/l_{f}.txt'.format(f = sigma), l, fmt = '%1.3f')
    np.savetxt('data/single_legged_hopper/erm/t.txt', t, fmt = '%1.3f')

# plot trajectory
plot_CC(base_height, foot_height, normal_force, t, sigmas)
print("Elapsed times: ", times)
print('Done!')
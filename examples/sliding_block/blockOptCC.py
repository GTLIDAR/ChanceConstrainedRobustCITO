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
from plotting_tool import plot_CC
# Create the block model with the default flat terrain
plant = TimeSteppingMultibodyPlant(file="systems/urdf/sliding_block.urdf")
plant.Finalize()
num_step = 101
step_size = 0.01
# Get the default context
context = plant.multibody.CreateDefaultContext()
# set chance constraints parameters
beta, theta = 0.6, 0.6
# sigmas = [0.01,0.05, 0.1, 0.3, 1]
sigmas = [0.01]
# sigmas = [1e-4, 1e-3, 1e-2, 1e-1, 1]
times = []
# set friction ERM parameters
friction_bias = 0.01
friction_multiplier = 1e6
# set uncertainty option
uncertainty_option = 3
# set chance constraint option
cc_option = 1
# Add initial and final state constraints
x0 = np.array([0., 0.5, 0., 0.])
xf = np.array([5., 0.5, 0., 0.])

# frictionVar = np.array([0.1])
# friction_variance = 0.03
# iterations 
iteration = len(sigmas)
# initialized saved trajectories
horizontal_position = np.zeros([iteration + 1, num_step])
horizontal_velocity = np.zeros([iteration + 1, num_step])
control = np.zeros([iteration, num_step])
friction = np.zeros([iteration, num_step])
# initial trajectory
x_init = np.loadtxt('data/slidingblock/warm_start/x.txt')
u_init = np.loadtxt('data/slidingblock/warm_start/u.txt')

l_init = np.loadtxt('data/slidingblock/warm_start/l.txt')
t = np.loadtxt('data/slidingblock/warm_start/t.txt')
# loop through different variance values
for i in range (iteration):
    sigma = sigmas[i]
    chance_params = np.array([beta, theta, sigma])
    friction_erm_params = np.array([sigma, friction_bias, friction_multiplier])
    # Create a Contact Implicit Trajectory Optimization
    trajopt = ChanceConstrainedContactImplicit(plant=plant,
                                            context=context,
                                            num_time_samples=num_step,
                                            maximum_timestep=step_size,
                                            minimum_timestep=step_size,
                                            chance_param= chance_params,
                                            friction_param= friction_erm_params,
                                            optionERM = uncertainty_option,
                                            optionCC = cc_option)
    trajopt.add_state_constraint(knotpoint=0, value=x0)    
    trajopt.add_state_constraint(knotpoint=100, value=xf)
    # Set all the timesteps to be equal
    trajopt.add_equal_time_constraints()
    # Add control and state costs
    # Add a running cost weights on the controls
    if i == 0:
        R = 100 * np.ones((1,1))
    else:
        R= 100 * np.ones((1,1))
    # R = 10 * np.ones((1,1))
    b = np.zeros((1,))
    # Add running cost weight on state
    Q = 1*np.diag([1,1,1,1])
    trajopt.add_quadratic_running_cost(R, b, [trajopt.u], name="ControlCost")
    trajopt.add_quadratic_running_cost(Q, xf, [trajopt.x], name="StateCost")
    # Add a final cost on the total time
    # cost = lambda h: np.sum(h)
    # trajopt.add_final_cost(cost, vars=[trajopt.h], name="TotalTime")
    u_init = u_init.reshape(trajopt.u.shape)
    # Set the initial trajectory guess, might switch out later for a warm start trajectory

    trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init)
    prog = trajopt.get_program()
    
    # Set the SNOPT solver options
    prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 1e7)
    
    if i == 1:
        prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-8)
        prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-8)
        prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 1)
    else:
        prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-8)
        prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-8)
        prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 2)
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
    t, xtraj = utils.GetKnotsFromTrajectory(x)
    fig1, axs1 = plt.subplots(3,1)
    axs1[0].plot(t,xtraj[0])
    plt.show()
    # horizontal_position[i, :] = x[0, :]
    # horizontal_velocity[i, :] = x[2, :]
    # control[i, :] = u[0, :]
    # friction[i, :] = l[1, :] - l[3,:]
# save trajectory
# np.savetxt('data/slidingblock/cc/horizontal_position.txt', horizontal_position, fmt = '%1.3f')
# np.savetxt('data/slidingblock/cc/control.txt', control, fmt = '%1.3f')
# np.savetxt('data/slidingblock/cc/friction.txt', friction, fmt = '%1.3f')
# np.savetxt('data/slidingblock/cc/t.txt', t, fmt = '%1.3f')
# plot trajectory
plot_CC(horizontal_position, control, friction, t, sigmas)
print("Elapsed times: ", times)
print('Done!')
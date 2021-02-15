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

# Create the block model with the default flat terrain
plant = TimeSteppingMultibodyPlant(file="systems/urdf/sliding_block.urdf")
plant.Finalize()
num_step = 101
step_size = 0.01
# Get the default context
context = plant.multibody.CreateDefaultContext()
# set chance constraints parameters
beta, theta, sigma = 0.6, 0.6, 0.2
chance_params = np.array([beta, theta, sigma])
# set friction ERM parameters
friction_bias = 0.01
friction_multiplier = 20
# set normal distance ERM parameters
distance_multiplier = 10
# set uncertainty option
uncertainty_option = 4
# Add initial and final state constraints
x0 = np.array([0., 0.5, 0., 0.])
xf = np.array([5., 0.5, 0., 0.])
# Add a running cost weights on the controls
R= 1 * np.ones((1,1))
b = np.zeros((1,))
# Add running cost weight on state
Q = 10*np.diag([1,1,1,1])
# Add a final cost on the total time
cost = lambda h: np.sum(h)
# iterations 
n = 1
# initialized saved trajectories
horizontal_position = np.zeros([n, num_step])
horizontal_velocity = np.zeros([n, num_step])
control = np.zeros([n,num_step])
# loop through different variance values
for i in range (n):
    # friction_variance = 0.1 + 0.1 * i
    # distance_variance = 0.1 + 0.1 * i
    friction_variance = 0.4
    distance_variance = 
    friction_erm_params = np.array([friction_variance, friction_bias, friction_multiplier])
    distance_erm_params = np.array([distance_variance, distance_multiplier])
    # Create a Contact Implicit Trajectory Optimization
    trajopt = ChanceConstrainedContactImplicit(plant=plant,
                                            context=context,
                                            num_time_samples=num_step,
                                            maximum_timestep=step_size,
                                            minimum_timestep=step_size,
                                            chance_param= chance_params,
                                            distance_param = distance_erm_params,
                                            friction_param= friction_erm_params,
                                            option = uncertainty_option)
    trajopt.add_state_constraint(knotpoint=0, value=x0)    
    trajopt.add_state_constraint(knotpoint=100, value=xf)
    # Set all the timesteps to be equal
    trajopt.add_equal_time_constraints()
    # Add control and state costs
    trajopt.add_quadratic_running_cost(R, b, [trajopt.u], name="ControlCost")
    trajopt.add_quadratic_running_cost(Q, xf, [trajopt.x], name="StateCost")
    # Set the initial trajectory guess, might switch out later for a warm start trajectory
    u_init = np.zeros(trajopt.u.shape)
    x_init = np.zeros(trajopt.x.shape)
    for n in range(0, x_init.shape[0]):
        x_init[n,:] = np.linspace(start=x0[n], stop=xf[n], num=101)
    l_init = np.zeros(trajopt.l.shape)
    trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init)
    prog = trajopt.get_program()
    # Set the SNOPT solver options
    prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 10000)
    prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-6)
    prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-6)
    prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 2)
    solver = SnoptSolver()
    # Solve the problem
    print("Solving trajectory optimization ", i)
    start = timeit.default_timer()
    result = solver.Solve(prog)
    stop = timeit.default_timer()
    print(f"Elapsed time: {stop-start}")
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
    horizontal_position[i, :] = x[0, :]
    horizontal_velocity[i, :] = x[2, :]
    control[i, :] = u[0, :]
# Plot the horizontal trajectory
fig1, axs1 = plt.subplots(3,1)
for i in range(n):
    axs1[0].plot(t, horizontal_position[i,:], linewidth=1.5)
    axs1[1].plot(t, horizontal_velocity[i,:], linewidth=1.5)
    axs1[2].plot(t, control[i,:], linewidth=1.5)
    plt.show()
axs1[0].set_title('Horizontal Trajectory')
axs1[0].set_ylabel('Position')
# axs1[0].set_xlim

axs1[1].set_ylabel('Velocity')


axs1[2].set_ylabel('Control (N)')
axs1[2].set_xlabel('Time (s)')

# Plot the reaction forces
fig3, axs3 = plt.subplots(3,1)
axs3[0].plot(t, l[0,:], linewidth=1.5)
axs3[0].set_ylabel('Normal')
axs3[0].set_title('Ground reaction forces')

axs3[1].plot(t, l[1,:] - l[3,:], linewidth=1.5)
axs3[1].set_ylabel('Friction-x')

axs3[2].plot(t, l[2, :] - l[4,:], linewidth=1.5)
axs3[2].set_ylabel('Friction-y')
axs3[2].set_ylim(-0.5, 3)
axs3[2].set_xlabel('Time (s)')

plt.show()
print('Done!')
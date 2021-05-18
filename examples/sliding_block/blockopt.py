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
from trajopt.contactimplicit import ContactImplicitDirectTranscription
from trajopt.robustContactImplicit import ChanceConstrainedContactImplicit
from systems.timestepping import TimeSteppingMultibodyPlant
from pydrake.solvers.snopt import SnoptSolver
from pydrake.math import RigidTransform, RollPitchYaw
import utilities as utils
from scipy.special import erfinv
import pickle
import os
from tempfile import TemporaryFile
from pydrake.all import PiecewisePolynomial
from systems.visualization import Visualizer

# Create the block model with the default flat terrain
_file = "systems/urdf/sliding_block.urdf"
plant = TimeSteppingMultibodyPlant(file= _file)
body_inds = plant.multibody.GetBodyIndices(plant.model_index)
base_frame = plant.multibody.get_body(body_inds[0]).body_frame()
plant.multibody.WeldFrames(plant.multibody.world_frame(), base_frame, RigidTransform())
plant.Finalize()
# Get the default context
context = plant.multibody.CreateDefaultContext()
# set chance constraints parameters
beta, theta, sigma = 0.6, 0.6, 0.3
chance_params = np.array([beta, theta, sigma])
# set friction ERM parameters
friction_variance = sigma

friction_bias = 0.01
friction_multiplier = 1e6
friction_erm_params = np.array([friction_variance, friction_bias, friction_multiplier])
# set normal distance ERM parameters
distance_variance = 0.1
distance_multiplier = 1e6
distance_erm_params = np.array([distance_variance, distance_multiplier])
# set uncertainty option
erm_option = 3
# set chance constraint option
cc_option = 1
# Create a Contact Implicit Trajectory Optimization
trajopt = ChanceConstrainedContactImplicit(plant=plant,
                                            context=context,
                                            num_time_samples=101,
                                            maximum_timestep=0.03,
                                            minimum_timestep=0.03,
                                            chance_param= chance_params,
                                            distance_param = distance_erm_params,
                                            friction_param= friction_erm_params,
                                            optionERM = erm_option,
                                            optionCC= cc_option)
# Add initial and final state constraints
x0 = np.array([0., 0.5, 0., 0.])
xf = np.array([5., 0.5, 0., 0.])
trajopt.add_state_constraint(knotpoint=0, value=x0)    
trajopt.add_state_constraint(knotpoint=100, value=xf)
# Set all the timesteps to be equal
trajopt.add_equal_time_constraints()
# trajopt.uncertainty_option = 4
# print(trajopt.uncertainty_option)

# Add a running cost on the controls
R= 10 * np.ones((1,1))
b = np.zeros((1,))

trajopt.add_quadratic_running_cost(R, b, [trajopt.u], name="ControlCost")

Q = 1*np.diag([1,1,1,1])
trajopt.add_quadratic_running_cost(Q, xf, [trajopt.x], name="StateCost")
# Add a final cost on the total time
# cost = lambda h: np.sum(h)
# trajopt.add_final_cost(cost, vars=[trajopt.h], name="TotalTime")

# Set the initial trajectory guess
u_init = np.zeros(trajopt.u.shape)
x_init = np.zeros(trajopt.x.shape)
for n in range(0, x_init.shape[0]):
    x_init[n,:] = np.linspace(start=x0[n], stop=xf[n], num=101)
l_init = np.zeros(trajopt.l.shape)

# load initial trajectories
# x_init = np.loadtxt('data/slidingblock/warm_start/x.txt')
# u_init = np.loadtxt('data/slidingblock/warm_start/u.txt')
# u_init = u_init.reshape(trajopt.u.shape)
# l_init = np.loadtxt('data/slidingblock/warm_start/l.txt')


# x_init = np.loadtxt('data/slidingblock/erm_cc_0.3/x.txt')
# u_init = np.loadtxt('data/slidingblock/erm_cc_0.3/u.txt')
# u_init = u_init.reshape(trajopt.u.shape)
# l_init = np.loadtxt('data/slidingblock/erm_cc_0.3/l.txt')
trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init)
# Get the final program, with all costs and constraints
prog = trajopt.get_program()
# Set the SNOPT solver options
prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 1e5)
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-6)
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-6)
prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 2)
solver = SnoptSolver()
# trajopt.enable_cost_display(display='figure')
# Check the problem for bugs in the constraints
if not utils.CheckProgram(prog):
    quit()

# Solve the problem
print("Solving trajectory optimization")
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
# # Save trajectory 
# np.savetxt('data/slidingblock/erm_cc_0.3/x.txt', x, fmt = '%1.3f')
# np.savetxt('data/slidingblock/erm_cc_0.3/u.txt', u, fmt = '%1.3f')
# np.savetxt('data/slidingblock/erm_cc_0.3/l.txt', l, fmt = '%1.3f')
# np.savetxt('data/slidingblock/erm_cc_0.3/t.txt', t, fmt = '%1.3f')

# Plot the horizontal trajectory
fig1, axs1 = plt.subplots(3,1)
axs1[0].plot(t, x[0,:], linewidth=1.5)
axs1[0].set_title('Horizontal Trajectory')
axs1[0].set_ylabel('Position')
# axs1[0].set_xlim
axs1[1].plot(t,x[2,:], linewidth=1.5)
axs1[1].set_ylabel('Velocity')

axs1[2].plot(t, u[0,:], linewidth=1.5)
axs1[2].set_ylabel('Control (N)')
axs1[2].set_xlabel('Time (s)')

# one collision point
fig3, axs3 = plt.subplots(3,1)
axs3[0].plot(t, l[0,:], linewidth=1.5)
axs3[0].set_ylabel('Normal')
axs3[0].set_title('Ground reaction forces')

axs3[1].plot(t, l[1,:] - l[3,:], linewidth=1.5)
axs3[1].set_ylabel('Friction-x')

axs3[2].plot(t, l[2, :] - l[4,:], linewidth=1.5)
axs3[2].set_ylabel('Friction-y')
# axs3[2].set_ylim(-0.5, 3)
axs3[2].set_xlabel('Time (s)')

# Show the plots
plt.show()
print('Done!')

x = PiecewisePolynomial.FirstOrderHold(t, x)
# x = PiecewisePolynomial.FirstOrderHold(t, x[:, 0])
vis = Visualizer(_file)
body_inds = vis.plant.GetBodyIndices(vis.model_index)
base_frame = vis.plant.get_body(body_inds[0]).body_frame()
vis.plant.WeldFrames(vis.plant.world_frame(), base_frame, RigidTransform())

vis.visualize_trajectory(x)

# Save the results
# file = "data/slidingblock/block_trajopt.pkl"
# data = trajopt.result_to_dict(result)
# utils.save(file, data)
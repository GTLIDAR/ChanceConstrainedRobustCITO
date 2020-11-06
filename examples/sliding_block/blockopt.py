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
from systems.timestepping import TimeSteppingMultibodyPlant
from pydrake.solvers.snopt import SnoptSolver
import utilities as utils

# Create the block model with the default flat terrain
plant = TimeSteppingMultibodyPlant(file="systems/urdf/sliding_block.urdf")
plant.Finalize()
# Get the default context
context = plant.multibody.CreateDefaultContext()
# Create a Contact Implicit Trajectory Optimization
trajopt = ContactImplicitDirectTranscription(plant=plant,
                                            context=context,
                                            num_time_samples=101,
                                            maximum_timestep=0.01,
                                            minimum_timestep=0.01)
# Add initial and final state constraints
x0 = np.array([0., 0.5, 0., 0.])
xf = np.array([5., 0.5, 0., 0.])
trajopt.add_state_constraint(knotpoint=0, value=x0)    
trajopt.add_state_constraint(knotpoint=100, value=xf)
# Set all the timesteps to be equal
trajopt.add_equal_time_constraints()
# Add a running cost on the controls
Q = 10 * np.ones((1,1))
b = np.zeros((1,))
trajopt.add_quadratic_running_cost(Q, b, [trajopt.u], name="ControlCost")
R = np.diag([1,1,1,1])
trajopt.add_quadratic_running_cost(R, xf, [trajopt.x], name="StateCost")
# Add a final cost on the total time
cost = lambda h: np.sum(h)
trajopt.add_final_cost(cost, vars=[trajopt.h], name="TotalTime")
# Set the initial trajectory guess
u_init = np.zeros(trajopt.u.shape)
x_init = np.zeros(trajopt.x.shape)
for n in range(0, x_init.shape[0]):
    x_init[n,:] = np.linspace(start=x0[n], stop=xf[n], num=101)
l_init = np.zeros(trajopt.l.shape)
trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init)
# Get the final program, with all costs and constraints
prog = trajopt.get_program()
# Set the SNOPT solver options
prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 10000)
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-6)
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-6)
prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 2)
solver = SnoptSolver()
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
# Plot the horizontal trajectory
fig1, axs1 = plt.subplots(3,1)
axs1[0].plot(t, x[0,:], linewidth=1.5)
axs1[0].set_title('Horizontal Trajectory')
axs1[0].set_ylabel('Position')
axs1[1].plot(t,x[2,:], linewidth=1.5)
axs1[1].set_ylabel('Velocity')

axs1[2].plot(t, u[0,:], linewidth=1.5)
axs1[2].set_ylabel('Control (N)')
axs1[2].set_xlabel('Time (s)')
# Plot the vertical trajectory, as a check
fig2, axs2 = plt.subplots(2,1)
axs2[0].plot(t, x[1,:], linewidth=1.5)
axs2[0].set_ylabel('Position')
axs2[0].set_title('Vertical Trajectory')

axs2[1].plot(t,x[3,:], linewidth=1.5)
axs2[1].set_ylabel('Velocity')
axs2[1].set_xlabel('Time (s)')
# Plot the reaction forces
fig3, axs3 = plt.subplots(3,1)
axs3[0].plot(t, l[0,:], linewidth=1.5)
axs3[0].set_ylabel('Normal')
axs3[0].set_title('Ground reaction forces')

axs3[1].plot(t, l[1,:] - l[3,:], linewidth=1.5)
axs3[1].set_ylabel('Friction-x')

axs3[2].plot(t, l[2, :] - l[4,:], linewidth=1.5)
axs3[2].set_ylabel('Friction-y')
axs3[2].set_xlabel('Time (s)')

# Show the plots
plt.show()
print('Done!')

# Save the results
file = "data/slidingblock/block_trajopt.pkl"
data = trajopt.result_to_dict(result)
utils.save(file, data)
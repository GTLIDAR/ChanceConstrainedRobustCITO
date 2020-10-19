"""
Contact Implicit Trajectory Optimization for a sliding block

Luke Drnach
October 15, 2020
"""
# Imports
import timeit
import numpy as np
import matplotlib.pyplot as plt
from trajopt.contactimplicit import ContactImplicitDirectTranscription
from systems.timestepping import TimeSteppingMultibodyPlant
from pydrake.solvers.snopt import SnoptSolver, SnoptSolverDetails
from utilities import CheckProgram


# Create the block model with the default flat terrain
plant = TimeSteppingMultibodyPlant(file="systems/urdf/sliding_block.urdf")
plant.Finalize()
# Get the default context
context = plant.multibody.CreateDefaultContext()
# Create a Contact Implicit Trajectory Optimization
trajopt = ContactImplicitDirectTranscription(plant=plant,
                                            context=context,
                                            num_time_samples=101,
                                            maximum_timestep=0.1,
                                            minimum_timestep=0.01)
# Add initial and final state constraints
x0 = np.array([0., 0.5, 0., 0., 0., 0.])
xf = np.array([5., 0.5, 0., 0., 0., 0.])
trajopt.add_state_constraint(knotpoint=0, value=x0)    
trajopt.add_state_constraint(knotpoint=100, value=xf)
# Set all the timesteps to be equal
trajopt.add_equal_time_constraints()
# Add a running cost on the controls
Q = 10 * np.ones((1,1))
b = np.zeros((1,))
# trajopt.add_quadratic_running_cost(Q, b, [trajopt.u], name="ControlCost")
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
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Iterations Limit", 1000)
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-4)
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-4)
prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 2)
solver = SnoptSolver()
# Check the problem for bugs in the constraints
#TODO: Re-write cost adders to force the output to a scalar type
if not CheckProgram(prog):
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
print(f"SNOPT Exit Status: {result.get_solver_details().info}")

# Unpack and plot the trajectories
x = trajopt.reconstruct_state_trajectory(result)
u = trajopt.reconstruct_input_trajectory(result)
l = trajopt.reconstruct_reaction_force_trajectory(result)
t = trajopt.get_solution_times(result)

# Plot the horizontal trajectory
plt.figure(1)
plt.title('Horizontal Trajectory')
plt.subplot(3,1,1)
plt.plot(t, x[0,:], linewidth=1.5)
plt.ylabel('Position')
plt.subplot(3,1,2)
plt.plot(t,x[3,:], linewidth=1.5)
plt.ylabel('Velocity')
plt.subplot(3,1,3)
plt.plot(t, u, linewidth=1.5)
plt.ylabel('Control (N)')
plt.xlabel('Time (s)')
# Plot the vertical trajectory, as a check
plt.figure(2)
plt.title('Vertical Trajectory')
plt.subplot(2,1,1)
plt.plot(t, x[1,:], linewidth=1.5)
plt.ylabel('Position')
plt.subplot(2,1,2)
plt.plot(t,x[4,:], linewidth=1.5)
plt.ylabel('Velocity')
plt.xlabel('Time (s)')
# Plot the reaction forces
plt.figure(3)
plt.title('Ground reaction forces')
plt.subplot(3,1,1)
plt.ylabel('Normal')
for n in range(0,4):
    plt.subplot(3,1,1)
    plt.plot(t, l[n,:], linewidth=1.5)
    plt.ylabel('Normal')
    plt.subplot(3,1,2)
    plt.plot(t, l[4(n+1),:] - l[4*(n+1)+2,:], linewidth=1.5)
    plt.ylabel('Friction-x')
    plt.subplot(3,1,3)
    plt.plot(t, l[4*(n+1)+1, :] - l[4*(n+1)+3,:], linewidth=1.5)
    plt.ylabel('Friction-y')
plt.xlabel('Time (s)')
# Show the plots
plt.show()
print('Done!')
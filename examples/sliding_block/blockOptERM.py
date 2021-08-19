# Imports
import timeit
import numpy as np
import matplotlib.pyplot as plt
# from trajopt.contactimplicit import ContactImplicitDirectTranscription
from trajopt.robustContactImplicit import OptimizationOptions, ChanceConstrainedContactImplicit
from systems.timestepping import TimeSteppingMultibodyPlant
from pydrake.solvers.snopt import SnoptSolver
import utilities as utils
from scipy.special import erfinv
import pickle
import os
from tempfile import TemporaryFile
from plotting_tool import plot_ERM
from trajopt.constraints import NCCImplementation, NCCSlackType
from pydrake.all import PiecewisePolynomial, RigidTransform
# Create the block model with the default flat terrain
# plant = TimeSteppingMultibodyPlant(file="systems/urdf/sliding_block.urdf")
_file = "systems/urdf/sliding_block.urdf"
plant = TimeSteppingMultibodyPlant(file= _file)
body_inds = plant.multibody.GetBodyIndices(plant.model_index)
base_frame = plant.multibody.get_body(body_inds[0]).body_frame()
plant.multibody.WeldFrames(plant.multibody.world_frame(), base_frame, RigidTransform())
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
friction_multiplier = 1e3
# set normal distance ERM parameters
distance_multiplier = 10
# set uncertainty option
uncertainty_option = 1
# set chance constraint option
cc_option = 1
# Add initial and final state constraints
x0 = np.array([0., 0.5, 0., 0.])
xf = np.array([5., 0.5, 0., 0.])
# Add a running cost weights on the controls
R= 10* np.ones((1,1))
b = np.zeros((1,))
# Add running cost weight on state
Q = 1*np.diag([1,1,1,1])
# Add a final cost on the total time
cost = lambda h: np.sum(h)
# frictionVar = np.array([0.1])
frictionVar = np.array([0.01, 0.03, 0.05, 0.1, 0.3])
distanceVar = np.array([0.01, 0.03, 0.05, 0.1, 0.3])
# iterations 
iteration = len(frictionVar)
# initialized saved trajectories
horizontal_position = np.zeros([iteration, num_step])
horizontal_velocity = np.zeros([iteration, num_step])
control = np.zeros([iteration,num_step])

# loop through different variance values
for i in range (iteration):
    friction_variance = frictionVar[i]
    distance_variance = distanceVar[i]
    friction_erm_params = np.array([friction_variance, friction_bias, friction_multiplier])
    distance_erm_params = np.array([distance_variance, distance_multiplier])
    # Create a Contact Implicit Trajectory Optimization
    options = OptimizationOptions()
    options.ncc_implementation = NCCImplementation.LINEAR_EQUALITY
    options.slacktype = NCCSlackType.CONSTANT_SLACK
    trajopt = ChanceConstrainedContactImplicit(plant=plant,
                                            context=context,
                                            num_time_samples=num_step,
                                            maximum_timestep=step_size,
                                            minimum_timestep=step_size,
                                            chance_param= chance_params,
                                            distance_param = distance_erm_params,
                                            friction_param= friction_erm_params,
                                            optionERM = uncertainty_option,
                                            optionCC = cc_option,
                                            options=options
                                            )
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
    print("Solving trajectory optimization ", i + 1)
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
    # s = trajopt.reconstruct_slack_trajectory(result)
    soln = trajopt.result_to_dict(result)
    _name = "erm_%d.pkl" %(sigma)
    _filename = "data/slidingblock/" + _name
    utils.save(_filename, soln)
    t, xtraj = utils.GetKnotsFromTrajectory(x)
    fig1, axs1 = plt.subplots(3,1)
    axs1[0].plot(t,xtraj[0,:])
    plt.show()
    # trajopt.plant_f.plot_trajectories(x, u, l)

    # horizontal_position[i, :] = x[0, :]
    # horizontal_velocity[i, :] = x[2, :]
    # control[i, :] = u[0, :]

# plot trajectory
# plot(horizontal_position, control, t, iteration, frictionVar)
print('Done!')
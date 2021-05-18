'''
This script is used to genertate all robust trajectories from
contact-implicit trajectory optimization, including ERM, ERM + Chance Constraint
for the legged hopper example
'''
# Imports
import timeit
import numpy as np
import matplotlib.pyplot as plt
from trajopt.contactimplicit import OptimizationOptions, ContactImplicitDirectTranscription 
from trajopt.robustContactImplicit import ChanceConstrainedContactImplicit
from systems.timestepping import TimeSteppingMultibodyPlant
from pydrake.solvers.snopt import SnoptSolver
from trajopt.constraints import NCCImplementation, NCCSlackType
import utilities as utils
from scipy.special import erfinv
import pickle
import os
from tempfile import TemporaryFile
from plot_hopper import plot_CC, plot_erm
from pydrake.all import PiecewisePolynomial, RigidTransform
import pickle
# Create the block model with the default flat terrain
_file = "systems/urdf/single_legged_hopper.urdf"
plant = TimeSteppingMultibodyPlant(file= _file)
body_inds = plant.multibody.GetBodyIndices(plant.model_index)
base_frame = plant.multibody.get_body(body_inds[0]).body_frame()
plant.multibody.WeldFrames(plant.multibody.world_frame(), base_frame, RigidTransform())
plant.Finalize()
num_step = 101
step_size = 0.03
# Get the default context
context = plant.multibody.CreateDefaultContext()
# set chance constraints parameters
# beta_theta = np.array([0.6, 0.85, 0.9])
# beta, theta = 0.5, 0.6
beta = 0.5
thetas = np.array([0.75 ])
# sigmas = [ 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
# sigmas = np.array([0.01, 0.05, 0.07, 0.09])
times = []
# set distance ERM parameters
distance_multiplier = 1e6
# set uncertainty option
uncertainty_option = 2
# set chance constraint option
cc_option = 2
# Add initial and final state
height = 1.5
angle = np.arccos(0.75/1)
x0 = np.array([0, height, -angle, 2*angle, -angle, 0, 0, 0, 0, 0])
xf = np.array([4, height, -angle, 2*angle, -angle, 0, 0, 0, 0, 0])
# iterations 
iteration = len(thetas)

# loop through different variance values
for i in range (iteration):
    # sigma = sigmas[i]
    sigma = 0.05
    # beta = beta_theta[i]
    theta = thetas[i]
    chance_params = np.array([beta, theta, sigma])
    distance_erm_params = np.array([sigma, distance_multiplier])
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
                                            distance_param= distance_erm_params,
                                            friction_param= np.array([0.01, 0.01, 1e5]),
                                            optionERM = uncertainty_option,
                                            optionCC = cc_option,
                                            options = options)
    init_num = '8'
    # load nominal traj as initial traj
    x_init = np.loadtxt('data/single_legged_hopper/nominal_3/x_{n}.txt'.format(n = init_num))
    u_init = np.loadtxt('data/single_legged_hopper/nominal_3/u_{n}.txt'.format(n = init_num))
    u_init = u_init.reshape(trajopt.u.shape)
    l_init = np.loadtxt('data/single_legged_hopper/nominal_3/l_{n}.txt'.format(n = init_num))
    jl_init = np.loadtxt('data/single_legged_hopper/nominal_3/jl_{n}.txt'.format(n = init_num))
    s_init = np.loadtxt('data/single_legged_hopper/nominal_3/s_{n}.txt'.format(n = init_num))
    t_init = np.loadtxt('data/single_legged_hopper/nominal_3/t.txt')
    trajopt.add_state_constraint(knotpoint=0, value=x0)    
    trajopt.add_state_constraint(knotpoint=100, value=xf)
    # Set all the timesteps to be equal
    trajopt.add_equal_time_constraints()
    # Add control and state costs
    # Add a running cost weights on the controls
    R=  0.01*np.diag([1,1,1])
    b = np.zeros((3,))
    # Add running cost weight on state
    Q = np.diag([1,10,10,100,100,1,1,1,1,1])
    trajopt.add_quadratic_running_cost(R, b, [trajopt.u], name="ControlCost")
    trajopt.add_quadratic_running_cost(Q, xf, [trajopt.x], name="StateCost")

    trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init, jltraj = jl_init, straj = s_init)
    prog = trajopt.get_program()
    
    # Set the SNOPT solver options
    prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 5*1e5)
    prog.SetSolverOption(SnoptSolver().solver_id(), 'Major Iterations Limit',5*1e4)
    prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-6)
    prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-6)
    prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option",2)
    solver = SnoptSolver()
    if not utils.CheckProgram(prog):
        quit()
    # Solve the problem
    print("Solving trajectory optimization, theta =", theta)
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
    save_num = 6
    folder = 'erm_cc_beta_theta'
    np.savetxt('data/single_legged_hopper/{f}/x_{n}{s}.txt'.format(n = save_num, s = theta, f = folder), x, fmt = '%1.3f')
    np.savetxt('data/single_legged_hopper/{f}/u_{n}{s}.txt'.format(n = save_num, s = theta, f = folder), u, fmt = '%1.3f')
    np.savetxt('data/single_legged_hopper/{f}/l_{n}{s}.txt'.format(n = save_num, s = theta, f = folder), l, fmt = '%1.3f')
    np.savetxt('data/single_legged_hopper/{f}/t.txt'.format(f = folder), t, fmt = '%1.3f')

# plot trajectory
# plot_CC(base_height, foot_height, normal_force, t, sigmas)
plot_erm(sigmas, folder ,save_num)
print("Elapsed times: ", times)
print('Done!')
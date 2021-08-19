# Imports
import timeit
import numpy as np
import matplotlib.pyplot as plt
# from pydrake.all import ClippingRange
# ClippingRange(0.01, 10.0)
# from pydrake.all import ClippingRange
from trajopt.contactimplicit import ContactImplicitDirectTranscription, OptimizationOptions
from trajopt.robustContactImplicit import ChanceConstrainedContactImplicit
from trajopt.constraints import NCCImplementation, NCCSlackType
from systems.timestepping import TimeSteppingMultibodyPlant
from pydrake.solvers.snopt import SnoptSolver
from systems.visualization import Visualizer
import utilities as utils
from plot_hopper import plot
from pydrake.all import PiecewisePolynomial, RigidTransform
import pickle
# Create the hopper model with the default flat terrain
_file = "systems/urdf/single_legged_hopper.urdf"
plant = TimeSteppingMultibodyPlant(file= _file)
body_inds = plant.multibody.GetBodyIndices(plant.model_index)
base_frame = plant.multibody.get_body(body_inds[0]).body_frame()
plant.multibody.WeldFrames(plant.multibody.world_frame(), base_frame, RigidTransform())
plant.Finalize()

options = OptimizationOptions()
options.ncc_implementation = NCCImplementation.LINEAR_EQUALITY
options.slacktype = NCCSlackType.CONSTANT_SLACK
# Get the default context
context = plant.multibody.CreateDefaultContext()
# set chance constraints parameters
beta, theta, sigma = 0.6, 0.6, 0.1
chance_params = np.array([beta, theta, sigma])
# set chance constraint option
cc_option = 1

# set normal distance ERM parameters
distance_variance = sigma
distance_multiplier = 1e6
distance_erm_params = np.array([distance_variance, distance_multiplier])
num_step = 101
# set uncertainty option
uncertainty_option = 1

# Create a Contact Implicit Trajectory Optimization
# trajopt = ContactImplicitDirectTranscription(plant=plant,
#                                                 context=context,
#                                                 num_time_samples=num_step,
#                                                 minimum_timestep = 0.03,
#                                                 maximum_timestep = 0.03,
#                                                 options = options)
trajopt = ChanceConstrainedContactImplicit(plant=plant,
                                            context=context,
                                            num_time_samples=num_step,
                                            maximum_timestep=0.03,
                                            minimum_timestep=0.03,
                                            chance_param= chance_params,
                                            distance_param= distance_erm_params,
                                            friction_param= np.array([0.01, 0.01, 1e5]),
                                            optionERM = uncertainty_option,
                                            optionCC = cc_option,
                                            options = options)
height = 1.5
angle = np.arccos(0.75/1)

# Add initial and final state
x0 = np.array([0, height, -angle, 2*angle, -angle, 0, 0, 0, 0, 0])
xf = np.array([4, height, -angle, 2*angle, -angle, 0, 0, 0, 0, 0])

trajopt.add_state_constraint(knotpoint=0, value=x0)    
trajopt.add_state_constraint(knotpoint=num_step - 1, value=xf)
# Set all the timesteps to be equal
trajopt.add_equal_time_constraints()
# Add a running cost on the controls
R=  0.01*np.diag([1,1,1])
b = np.zeros((3,))

trajopt.add_quadratic_running_cost(R, b, [trajopt.u], name="ControlCost")

Q = np.diag([1,10,10,100,100,1,1,1,1,1])
trajopt.add_quadratic_running_cost(Q, xf, [trajopt.x], name="StateCost")
# Add a final cost on the total time

# Set the initial trajectory guess
# u_init = np.zeros(trajopt.u.shape)
# x_init = np.zeros(trajopt.x.shape)

# l_init = np.zeros(trajopt.l.shape)
# data = utils.load('data/single_legged_hopper/reference_traj_mat.pkl')
# x_init = data['state'][:]
# u_init = data['control'][:] 
# t_init = data['time'][:]
# l_init = data['force'][:]
# s_init = data['slacks'][:]
# jl_init = data['jointlimit'][:]
# load initial traj
init_num = '8'
x_init = np.loadtxt('data/single_legged_hopper/nominal_3/x_{n}.txt'.format(n = init_num))
u_init = np.loadtxt('data/single_legged_hopper/nominal_3/u_{n}.txt'.format(n = init_num))
u_init = u_init.reshape(trajopt.u.shape)
l_init = np.loadtxt('data/single_legged_hopper/nominal_3/l_{n}.txt'.format(n = init_num))
jl_init = np.loadtxt('data/single_legged_hopper/nominal_3/jl_{n}.txt'.format(n = init_num))
s_init = np.loadtxt('data/single_legged_hopper/nominal_3/s_{n}.txt'.format(n = init_num))
t_init = np.loadtxt('data/single_legged_hopper/nominal_3/t.txt')
trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init, jltraj = jl_init, straj = s_init)
# slack = 0
# trajopt.set_slack(slack)
# Get the final program, with all costs and constraints
prog = trajopt.get_program()
# Set the SNOPT solver options
prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 1e5)
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-6)
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-6)
prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 2)
solver = SnoptSolver()
# trajopt.set_slack(10)
# trajopt.enable_cost_display(display='figure')
if not utils.CheckProgram(prog):
    quit()

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
# joint_force = trajopt.reconstruct_limit_force_trajectory(result).transpose()
jl= result.GetSolution(trajopt.jl)
s = result.GetSolution(trajopt.slacks)
# plt.plot(t, joint_force)

# with open('data/single_legged_hopper/reference_traj_py.pkl', 'wb') as f:
#     pickle.dump({'state':x,'control':u,'time':t, 'force': l, 'slacks': s, 'jointlimit': jl},f)
plot(x, u, l, t)
save_num = 'optimal8'
np.savetxt('data/single_legged_hopper/nominal_3/x_{n}.txt'.format(n = save_num), x, fmt = '%1.3f')
np.savetxt('data/single_legged_hopper/nominal_3/u_{n}.txt'.format(n = save_num), u, fmt = '%1.3f')
np.savetxt('data/single_legged_hopper/nominal_3/l_{n}.txt'.format(n = save_num), l, fmt = '%1.3f')
np.savetxt('data/single_legged_hopper/nominal_3/jl_{n}.txt'.format(n = save_num), jl, fmt = '%1.3f')
np.savetxt('data/single_legged_hopper/nominal_3/s_{n}.txt'.format(n = save_num), s, fmt = '%1.3f')
np.savetxt('data/single_legged_hopper/nominal_3/t.txt', t, fmt = '%1.3f')

# x = PiecewisePolynomial.FirstOrderHold(t, x)

# vis = Visualizer(_file)
# body_inds = vis.plant.GetBodyIndices(vis.model_index)
# base_frame = vis.plant.get_body(body_inds[0]).body_frame()
# vis.plant.WeldFrames(vis.plant.world_frame(), base_frame, RigidTransform())

# vis.visualize_trajectory(x)
print('Done!')
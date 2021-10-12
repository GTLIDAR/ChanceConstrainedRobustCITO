# ChanceConstrainedRobustCITO
Repository for Chance-Constrained Robust Contact-Implicit Trajectory Optimization.
This is an extension from our previous work on [RobustTO](http://lab-idar.gatech.edu/wp-content/uploads/Publications/robust-traj-opt-2021.pdf), implemented [here](https://github.com/GTLIDAR/RobustContactERM).

**[March 2021]** The [research abstract](http://lab-idar.gatech.edu/wp-content/uploads/Publications/DW2021_Chance_Constraint.pdf) was submitted to [Dynamic Walking](https://www.dynamicwalking2021.org/) workshop. 

**[May 2021]** The paper titled [*Mediating between Contact Feasibility and Robustness of Trajectory Optimization through Chance Complementarity Constraints*](http://lab-idar.gatech.edu/wp-content/uploads/Publications/Chance_Constrained_Robust_CITO_2021.pdf) was submitted to both [Frontiers in Robotics & AI](https://www.frontiersin.org/research-topics/25532/advancements-in-trajectory-optimization-and-model-predictive-control-for-legged-systems) and arXiv

## Overview
This respository is an implementation of the chance constrained robust contact-implicity trajectory optimization. Our work uses chance constraints as a probability-based constraint relaxation method and Expected Residual Minimization (ERM) as a smoothing method for solving complementarity problems with uncertain contact characteristics. Our method is tested on a bench marking push block over terrain with uncertain friction (see chance_block branch) and a footed hopper robot over terrain with uncertain contact distance (see chance_hopper branch). 

* Authors: Luke Drnach*, [John Z. Zhang*](https://jzzhang3.github.io/), and Ye Zhao (*Authors contributed equally)
* Affiliation: [The LIDAR Lab](http://lab-idar.gatech.edu/), Georgia Institute of Technology

## Installation
Clone the repository: `git clone https://github.com/GTLIDAR/ChanceConstrainedRobustCITO.git`

Install the Docker container release of Drake with VSCode following instructions [here](https://drake.mit.edu/docker.html). Every script is run in the container. 

## Tutorial
An example can be found in: `./examples/sliding_block/blockOpt.py`. This script solves the trajectory optimization problem in which a cube of 1m, 1kg is pushed 5m in 1s with quadratic control and state costs. The following is a tutorial for setting up and solving the sliding block example, similar techniques can be applied to the single legged hopper and other robot models. 

Initialize plant with flat terrain and create context:

```
_file = "systems/urdf/sliding_block.urdf"
plant = TimeSteppingMultibodyPlant(file= _file)
body_inds = plant.multibody.GetBodyIndices(plant.model_index)
base_frame = plant.multibody.get_body(body_inds[0]).body_frame()
plant.multibody.WeldFrames(plant.multibody.world_frame(), base_frame, RigidTransform())
plant.Finalize()
context = plant.multibody.CreateDefaultContext()
```
Create Contact-Implicit Trajectory Optimization object with uncertainty and chance constraint settings:
```
trajopt = ChanceConstrainedContactImplicit(plant=plant,
                                            context=context,
                                            num_time_samples=101,
                                            maximum_timestep=0.01,
                                            minimum_timestep=0.01,
                                            chance_param= np.array([0.6, 0.6, 0.1]),
                                            distance_param = np.array([0.1, 1e6]),
                                            friction_param= np.array([0.1, 0.01, 1e6]),
                                            optionERM = 1,
                                            optionCC= 1)
```
where chance constraints parameters, *chance_param* are *\beta, \theta, \sigma*, distance ERM parameters *distance_param* *are *sigma, ERM multiplier*, friction cone ERM paramters *friction_param* are *sigma, bias, ERM multiplier*. Note that distance and friction ERM multipliers do not have to match. 

Uncertainty Options:
```
#   option 1: no uncertainty
#   option 2: uncertainty from normal distance
#   option 3: unvertainty from friction cone
```

Chance Constraint Relaxation Options:
```
#   option 1: strict contact constraints
#   option 2: chance constraint relaxation for normal distance
#   option 3: chance constraint relaxation for friction cone
```

Add initial and final states:
```
x0 = np.array([0., 0.5, 0., 0.])
xf = np.array([5., 0.5, 0., 0.])
trajopt.add_state_constraint(knotpoint=0, value=x0)    
trajopt.add_state_constraint(knotpoint=100, value=xf)
```
Add quadratic running cost functions on state and control:
```
R= 10 * np.ones((1,1))
b = np.zeros((1,))
trajopt.add_quadratic_running_cost(R, b, [trajopt.u], name="ControlCost")
Q = 1*np.diag([1,1,1,1])
trajopt.add_quadratic_running_cost(Q, xf, [trajopt.x], name="StateCost")
```
Add initial trajectory guess:
```
u_init = np.zeros(trajopt.u.shape)
x_init = np.zeros(trajopt.x.shape)
for n in range(0, x_init.shape[0]):
    x_init[n,:] = np.linspace(start=x0[n], stop=xf[n], num=101)
l_init = np.zeros(trajopt.l.shape)
trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init)
```
A pre-solved warm-start trajectory can also be used as the initial guess. 

Set SNOPT solver settings:
```
prog = trajopt.get_program()
prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 1e5)
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-6)
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-6)
prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 2)
solver = SnoptSolver()
```
Solve the mathematical program:
```
start = timeit.default_timer()
result = solver.Solve(prog)
stop = timeit.default_timer()
print(f"Elapsed time: {stop-start}")
```
Unpack result trajectories:
```
x = trajopt.reconstruct_state_trajectory(result)
u = trajopt.reconstruct_input_trajectory(result)
l = trajopt.reconstruct_reaction_force_trajectory(result)
t = trajopt.get_solution_times(result)
```
Visualize the results in MeshCat:
```
x = PiecewisePolynomial.FirstOrderHold(t, x)
vis = Visualizer(_file)
body_inds = vis.plant.GetBodyIndices(vis.model_index)
base_frame = vis.plant.get_body(body_inds[0]).body_frame()
vis.plant.WeldFrames(vis.plant.world_frame(), base_frame, RigidTransform())
vis.visualize_trajectory(x)
```


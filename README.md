# ChanceConstrainedRobustCITO
Repository for Chance-Constrained Robust Contact-Implicit Trajectory Optimization.
This is an extension from our previous work on [RobustTO](http://lab-idar.gatech.edu/wp-content/uploads/Publications/robust-traj-opt-2021.pdf), implemented [here](https://github.com/GTLIDAR/RobustContactERM).

**[March 2021]** The [research abstract](http://lab-idar.gatech.edu/wp-content/uploads/Publications/DW2021_Chance_Constraint.pdf) was submitted to [Dynamic Walking](https://www.dynamicwalking2021.org/) workshop. 

**[May 2021]** The paper was submitted to both [L-CSS](http://ieee-cssletters.dei.unipd.it/index.php) and [arXiv]()

## Overview
This respository is an implementation of the chance constrained robust contact-implicity trajectory optimization. Our work uses chance constraints as a probability-based constraint relaxation method and Expected Residual Minimization (ERM) as a smoothing method for solving complementarity problems with uncertain contact characteristics. Our method is tested on a bench marking push block over terrain with uncertain friction and a footed hopper robot over terrain with uncertain contact distance. 

* Authors: [John Z. Zhang](jzzhang3.github.io), Luke Drnach, and Ye Zhao
* Affiliation: [The LIDAR Lab](http://lab-idar.gatech.edu/), Georgia Institute of Technology

## Installation
Clone the repository: `git clone https://github.com/GTLIDAR/ChanceConstrainedRobustCITO.git`

Install the Docker container release of Drake with VSCode following instructions [here](https://drake.mit.edu/docker.html). Every script is run in the container. 

## Tutorial
An example can be found in: `./examples/sliding_block/blockOpt.py`. This script solves the trajectory optimization problem in which a cube of 1m, 1kg is pushed 5m in 1s with quadratic control and state costs. 

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
where #chance_param = $\beta$, $\theta$, $\sigma$#, 
To run the single legged hopper example, run file: `./examples/single_legged_hopper/HopperOptNominal.py`


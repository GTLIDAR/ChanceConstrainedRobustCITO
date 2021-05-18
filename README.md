# ChanceConstrainedCITO
Repository for Chance-Constrained Robust Contact-Implicit Trajectory Optimization.
This is an extension from our previous work on [RobustTO](http://lab-idar.gatech.edu/wp-content/uploads/Publications/robust-traj-opt-2021.pdf), implemented [here](https://github.com/GTLIDAR/RobustContactERM).

**[March 2021]** The [research abstract](http://lab-idar.gatech.edu/wp-content/uploads/Publications/DW2021_Chance_Constraint.pdf) was submitted to [Dynamic Walking](https://www.dynamicwalking2021.org/) workshop. 

**[May 2021]** The paper was submitted to both [L-CSS](http://ieee-cssletters.dei.unipd.it/index.php) and [arXiv]()

## Overview
This respository is an implementation of the chance constrained robust contact-implicity trajectory optimization. Our work uses chance constraints as a probability-based constraint relaxation method and Expected Residual Minimization (ERM) as a smoothing method for solving complementarity problems with uncertain contact characteristics. Our method is tested on a bench marking push block over terrain with uncertain friction and a footed hopper robot over terrain with uncertain contact distance. 

* Authors: John Z. Zhang, Luke Drnach, and Ye Zhao
* Affiliation: [The LIDAR Lab](http://lab-idar.gatech.edu/), Georgia Institute of Technology

This code was tested in Python 3.6.9 and Docker container release of [Drake](https://drake.mit.edu/docker.html). 
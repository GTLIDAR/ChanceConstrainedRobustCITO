## PYCITO: Contact Implicit Trajectory Optimization in Python, using pyDrake

This project is currently in development. 

### Overview
This project contains utilities for robot motion planning with intermittent frictional contact. We use pyDrake as a backend for calculating rigid body dynamics and for solving general nonlinear programming problems. We also use and extend the trajectory optimization tools and implement contact-implicit trajectory optimization in Python.

+ Author: Luke Drnach
+ Affiliation: [The LIDAR Lab](http://lab-idar.gatech.edu/), Georgia Institute of Technology

### pyDrake tutorials
To better understand the interface with pyDrake, pyCITO contains several tutorials on Drake's MultibodyPlant and MathematicalProgram classes. The *tutorials* directory contains several tutorials on the Drake methods available in pyDrake and how to use them. 

In *tutorials/multibody*:
+ **kinematics.py** details accessing rigid body frames and performing forward kinematics using MultibodyPlant
+ **dynamics.py** details accessing the dynamic properties - such as the generalized mass and coriolis matrices - from MultibodyPlant, setting external forces, and performing inverse dynamics
+ **collision.py** details extracting collision geometries from MultibodyPlant

In *tutorials/optimization*:
+ **mathprog.py** details setting up a simple optimization problem with Drake's MathematicalProgram
+ **doubleIntegrator.py** details setting up a trajectory optimization problem on a linear system with custom constraints, using MathematicalProgram
+ **acrobotSwingUp.py** details setting up a trajectory optimization problem using MultibodyPlant and DirectCollocation
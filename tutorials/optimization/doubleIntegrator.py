"""
Example of adding a custom constraint to a mathematical program in pydrake.

This example solves a trajectory optimization problem with a double integrator. The problem is constructed using MathematicalProgram, instead of using DirectCollocation, to highlight several features in MathematicalProgram.

The goal is to drive the double integrator from a non-zero initial state to the origin. No cost is placed on the controls or state. 

This example is adapted from "Trajectory optimization for the double integrator" in Russ Tedrake's Underactuated Robotics course: http://underactuated.mit.edu/trajopt.html

Luke Drnach
October 26, 2020
"""

import numpy as np 
import matplotlib.pyplot as plt 
from pydrake.all import MathematicalProgram, Solve, Variable, eq

# Approximate the double integrator
dt = 0.01
A = np.eye(2) + dt*np.array([[0, 1],[0,0]])
B = dt * np.array([0,1]).T
B = np.expand_dims(B, axis=1)
# Dynamics constraint function
def dynamicsCstr(z):
    x1, u1, y = np.split(z, [2,3])
    return y - A.dot(x1) - B.dot(u1)

# Create a  mathematical program
prog = MathematicalProgram()
# Number of knot points
N = 284

# Create decision variables
u = np.empty((1,N-1), dtype=Variable)
x = np.empty((2,N), dtype=Variable)
for n in range(0, N-1):
    u[:,n] = prog.NewContinuousVariables(1, 'u' + str(n))
    x[:,n] = prog.NewContinuousVariables(2, 'x' + str(n))
x[:,N - 1] = prog.NewContinuousVariables(2, 'x' + str(N))

# Add constraints at every knot point
x0 = [-2, 0]
prog.AddBoundingBoxConstraint(x0, x0, x[:,0])
for n in range(0, N-1):
    # Add the dynamics as an equality constraint
    # prog.AddConstraint(eq(x[:,n+1], A.dot(x[:,n]) + B.dot(u[:,n])))
    # Add the dynamics as a function handle constraint
    prog.AddConstraint(dynamicsCstr, lb=np.array([0., 0.]), ub=np.array([0., 0.]), vars=np.concatenate((x[:,n], u[:, n], x[:, n+1]), axis=0), description="dynamics")
    prog.AddBoundingBoxConstraint(-1, 1, u[:,n])
xf = [0, 0]
prog.AddBoundingBoxConstraint(xf, xf, x[:,N - 1])

# Solve the problem
result = Solve(prog)

x_sol = result.GetSolution(x)
print(f"Optimization successful? {result.is_success()}")

# Display the optimized trajectories
plt.figure()
plt.plot(x_sol[0,:], x_sol[1,:])
plt.xlabel('q')
plt.ylabel('qdot')
plt.show()
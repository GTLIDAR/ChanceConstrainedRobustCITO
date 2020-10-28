"""
kuka_custom: Example script on adding custom functions to optimization problems, using the Kuka manipulator as an example.

This optimization solves the inverse-kinematics problem for the Kuka manipulator arm. The IK objective is formulated as a cost and added to the problem. 

This example highlights adding costs and constraints that rely on MulitbodyPlant to a MathematicalProgram. Such constraints must be implemented for MultibodyPlants based on both the AutoDiffXd-type and the float type. 

Adapted from "Mathematical Program Multibody Tutorial" on the MIT Drake Website https://drake.mit.edu

Luke Drnach
October 14, 2020
"""
import numpy as np 
from utilities import CheckProgram
from pydrake.common import FindResourceOrThrow
from pydrake.math import RigidTransform 
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator 
from pydrake.all import MultibodyPlant
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve

# Create the MultibodyPlant Model
plant_f = MultibodyPlant(0.0)
iiwa_file = FindResourceOrThrow("drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf")
iiwa = Parser(plant_f).AddModelFromFile(iiwa_file)
# Get references to important frames
W = plant_f.world_frame()
L0 = plant_f.GetFrameByName("iiwa_link_0", iiwa)
L7 = plant_f.GetFrameByName("iiwa_link_7", iiwa)
# Weld the 0th frame to the world frame to fix the Kuka base
plant_f.WeldFrames(W, L0)
# Finalize the model
plant_f.Finalize()
# To use MultibodyPlant in a custom cost or constraint, we must implement the constraint for both float types and AutoDiffXd types. We also need references to both types of plant and context
# Create float context
context_f = plant_f.CreateDefaultContext()
# Create AutoDiffXd plant and context
plant_d = plant_f.ToAutoDiffXd()
context_d = plant_d.CreateDefaultContext()

# Define the target position of the kuka arm
p_WT = [0.1, 0.1, 0.6]

# Helper function
def resolve_frame(plant, F):
    """Get a frame from a plant, regardless of scalar type"""
    return plant.GetFrameByName(F.name(),F.model_instance()) 

# check how many times each instance of the cost is calculated
float_calls = 0
autodiff_calls = 0

# Custom cost function
def link_7_distance_to_target(q):
    """Squared distance between L7 origin and target position"""
    # Choose the plant and context based on type
    global float_calls
    global autodiff_calls
    if q.dtype == "float":
        plant = plant_f
        context = context_f
        float_calls+=1
    else:
        plant = plant_d
        context = context_d
        autodiff_calls+=1
    # Do forward kinematics
    plant.SetPositions(context, iiwa, q)
    X_WL7 = plant.CalcRelativeTransform(context, resolve_frame(plant, W), resolve_frame(plant, L7))
    p_TL7 = X_WL7.translation() - p_WT
    # Return scalar squared distance
    return p_TL7.dot(p_TL7)

# NOTE: Returning a scalar for a constraint or a vector for a cost could result in cryptic errors

# Create a mathematical program
prog = MathematicalProgram()
q = prog.NewContinuousVariables(plant_f.num_positions())
# define nominal configuration
q0 = np.zeros(plant_f.num_positions())
# Add the custom cost
prog.AddCost(link_7_distance_to_target, vars=q, description="IK_Cost")
# Solve the problem
CheckProgram(prog)
print('Start optimization')
result = Solve(prog, initial_guess=q0)
# Number of autodiff and float calls during optimization
print(f"Number float evaluations: {float_calls}")
print(f"Number of autodiff evaluations: {autodiff_calls}")

print(f"Success? {result.is_success()}")
print(result.get_solution_result())
qsol = result.GetSolution(q)
print(qsol)
print(f"Initial distance to target: {link_7_distance_to_target(q0):.3f}")
print(f"Final distance to target: {link_7_distance_to_target(qsol):.3f}")

# NOTE (luke): It appears that the optimization does NOT require implementing both the autodiff and float versions of the cost. Running the optimization solely uses the AUTODIFF version. HOWEVER, if we later want to use the cost function to evaluate the solution, we need the FLOAT version as well. 
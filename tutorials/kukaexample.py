import numpy as np 
from pydrake.common import FindResourceOrThrow
from pydrake.math import RigidTransform 
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator 
from pydrake.all import MultibodyPlant
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve

plant_f = MultibodyPlant(0.0)
iiwa_file = FindResourceOrThrow("drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf")
iiwa = Parser(plant_f).AddModelFromFile(iiwa_file)

print("success")
print(iiwa)
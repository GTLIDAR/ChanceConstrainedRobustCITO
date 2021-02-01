"""
Description of A1 robot

Luke Drnach
November 5, 2020
"""

from utilities import FindResource
from pydrake.all import MultibodyPlant
from pydrake.multibody.parsing import Parser

def create_a1_multibody():
    file = "systems/A1/A1_description/urdf/a1.urdf"
    plant = MultibodyPlant(0.0)
    a1 = Parser(plant).AddModelFromFile(FindResource(file))
    plant.Finalize()
    return(plant, a1)

def create_a1_timestepping():
    pass


if __name__ == "__main__":
    plant, _ = create_a1_multibody()
    print(f"A1 has {plant.num_positions()} position variables and {plant.num_velocities()} velocity variables")
    print(f"A1 has {plant.num_actuators()} actuators")
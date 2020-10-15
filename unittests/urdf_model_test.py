"""
Tests expected implementation of urdf models

Luke Drnach
October 14, 2020
"""

import numpy as np
import unittest
from utilities import FindResource
from pydrake.all import MultibodyPlant, RigidTransform
from pydrake.multibody.parsing import Parser

class TestSlidingBlock(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Find and load the sliding_block urdf file"""
        urdf_file = FindResource("systems/urdf/sliding_block.urdf")
        cls.plant = MultibodyPlant(0.0)
        cls.block = Parser(cls.plant).AddModelFromFile(urdf_file)
        body_inds = cls.plant.GetBodyIndices(cls.block)
        base_frame = cls.plant.get_body(body_inds[0]).body_frame()
        world_frame = cls.plant.world_frame()
        cls.plant.WeldFrames(world_frame, base_frame, RigidTransform())
        cls.plant.Finalize()

    def setUp(self):
        """Create a context variable for each test"""
        self.context = self.plant.CreateDefaultContext()

    def test_num_positions(self):
        """Check the number of position variables in the model"""
        self.assertEqual(self.plant.num_positions(), 3, msg="Wrong number of position variables")

    def test_num_velocities(self):
        """Check the number of velocity variables in the model"""
        self.assertEqual(self.plant.num_velocities(), 3, msg="Wrong number of velocity variables")

    def test_num_actuators(self):
        """Check the number of actuators in the model"""
        self.assertEqual(self.plant.num_actuators(), 1, msg="Wrong number of actuators")

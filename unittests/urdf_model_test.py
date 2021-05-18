"""
Tests expected implementation of urdf models

Luke Drnach
October 14, 2020
"""

import numpy as np
import unittest
from math import pi, sin, cos
from utilities import FindResource
from pydrake.all import MultibodyPlant, RigidTransform, JacobianWrtVariable, MultibodyForces
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
        x = [0,3, pi/4, 1, -0.5, 0.01]
        self.plant.SetPositionsAndVelocities(self.context, x)

    def test_num_positions(self):
        """Check the number of position variables in the model"""
        self.assertEqual(self.plant.num_positions(), 3, msg="Wrong number of position variables")

    def test_num_velocities(self):
        """Check the number of velocity variables in the model"""
        self.assertEqual(self.plant.num_velocities(), 3, msg="Wrong number of velocity variables")

    def test_num_actuators(self):
        """Check the number of actuators in the model"""
        self.assertEqual(self.plant.num_actuators(), 1, msg="Wrong number of actuators")

    def test_mass_matrix(self):
        """ Check the value of the mass matrix"""
        M = self.plant.CalcMassMatrixViaInverseDynamics(self.context)
        M_true = np.diag([1., 1., 0.167])
        self.assertTrue(np.allclose(M,M_true,), msg="Mass matrix is incorrect")

    def test_bias_term(self):
        """Check the value of the Coriolis and centrifugal terms"""
        C = self.plant.CalcBiasTerm(self.context)
        C_true = np.zeros((3,))
        self.assertTrue(np.allclose(C, C_true), msg="Bias term is incorrect")

    def test_grav_term(self):
        """Check the value of gravitational and conservative forces"""
        G = self.plant.CalcGravityGeneralizedForces(self.context)
        G_true = np.zeros((3,))
        G_true[1] = -9.81
        self.assertTrue(np.allclose(G,G_true), msg="Gravity terms is incorrect")

    def test_actuation_matrix(self):
        """Test the actuation matrix"""
        B = self.plant.MakeActuationMatrix()
        B_true = np.array([[1.], [0.], [0.]])
        self.assertTrue(np.allclose(B, B_true), msg="Actuation matrix is incorrrect")

    def test_inverse_dynamics(self):
        """Check the value returned by inverse dynamics"""
        dv = np.array([0.1, -0.1, 0.2])
        m = np.array([1, 1, 0.167])
        # Note that inverse dynamics does not account for gravity
        tau_true = m * dv
        mbf = MultibodyForces(self.plant)
        tau = self.plant.CalcInverseDynamics(self.context, dv, mbf)
        self.assertTrue(np.allclose(tau, tau_true), msg="Inverse dynamics is incorrect")

    def test_center_jacobian(self):
        """Check the value returned by the Jacobian"""
        p = np.array([0., 0., -0.5])
        q = self.plant.GetPositions(self.context)
        Jp_true = np.array([[1.0, 0.0, -0.5*cos(q[2])],
                            [0.0, 0.0,  0.0],
                            [0.0, 1.0,  0.5*sin(q[2])]])
        base_frame = self.plant.GetBodyByName('box').body_frame()
        wframe = self.plant.world_frame()
        Jp = self.plant.CalcJacobianTranslationalVelocity(self.context, JacobianWrtVariable.kQDot, base_frame, p, wframe, wframe)
        self.assertTrue(np.allclose(Jp, Jp_true), msg="Jacobian is incorrect")
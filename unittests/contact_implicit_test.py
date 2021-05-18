"""
unittests for contactimplicit.py

Luke Drnach
October 14, 2020
"""

import numpy as np
import unittest
from trajopt.contactimplicit import ContactImplicitDirectTranscription
from systems.timestepping import TimeSteppingMultibodyPlant
from pydrake.autodiffutils import AutoDiffXd

class ContactImplicitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup and finalize the plant model, and create the optimization problem"""
        cls.model = TimeSteppingMultibodyPlant(file="systems/urdf/sliding_block.urdf")
        cls.model.Finalize()
        cls.opt = ContactImplicitDirectTranscription(plant=cls.model,
                                                context=cls.model.multibody.CreateDefaultContext(),
                                                num_time_samples=101,
                                                minimum_timestep=0.001,
                                                maximum_timestep=0.1)
    
    def setUp(self):
        """Defines some dummy variables for use in each constraint evaluation"""
        self.h = np.array([0.01])
        self.u = np.array([2.5])
        self.x1 = np.array([0.0, 1.0, 0.1, 0.5])
        self.x2 = np.array([1.0, 0.5, 0.2, 0.1])
        self.l = np.array([10.0, 1.5, 2.1, 0.5, 3.7, 4.2])
        self.numN = 1
        self.numT = 4

    def test_opt_creation(self):
        """Check that the optimization can be set up"""        
        # First check that the object is not none
        self.assertIsNotNone(self.opt, msg="Optimization creation returned None")
        
    def test_eval_dynamic_constraint(self):
        """Check that the dynamic constraint can be evaluated"""
        z = np.concatenate([self.h, self.x1, self.x2, self.u, self.l], axis=0)
        r = self.opt._ContactImplicitDirectTranscription__backward_dynamics(z)
        r_true = np.array([0.998, -0.501, 6.5, -40.19])
        self.assertTrue(np.allclose(r,r_true), msg="Backward dynamics incorrect")
        
    def test_eval_normaldist_constraint(self):
        """Check the normal distance constraint"""
        # Check that the constraint returns the correct values for non-contact
        z = np.concatenate([self.x1, self.l[0:self.numN]], axis=0)
        r = self.opt._ContactImplicitDirectTranscription__normal_distance_constraint(z)
        phi = self.x1[1] - 0.5
        cstr_true = np.array([phi, self.l[0], phi*self.l[0]])
        self.assertTrue(np.allclose(r, cstr_true), msg="Normal distance constraint incorrect for no-contact")
        # Check that the constraint returns the correct values for contact
        z = np.concatenate([self.x2, self.l[0:self.numN]], axis=0)
        r = self.opt._ContactImplicitDirectTranscription__normal_distance_constraint(z)
        cstr_true = np.array([0.0, self.l[0], 0.0])
        self.assertTrue(np.allclose(r, cstr_true), msg="Normal distance constraint incorrect for contact")

    def test_eval_slidingvel_constraint(self):
        """Check that the sliding velocity constraint can be evaluated"""
        z = np.concatenate([self.x1,self.l[self.numN:]], axis=0)
        r = self.opt._ContactImplicitDirectTranscription__sliding_velocity_constraint(z)
        r_true = np.array([4.3, 4.2, 4.1, 4.2, 1.5, 2.1, 0.5, 3.7, 6.45, 8.82, 2.05, 15.54])
        self.assertTrue(np.allclose(r, r_true), msg = "Sliding velocity constraint incorrect")

    def test_eval_friccone_constraint(self):
        """Check that the friction cone constraint can be evaluated"""
        z = np.concatenate([self.x1, self.l], axis=0)
        r = self.opt._ContactImplicitDirectTranscription__friction_cone_constraint(z)
        r_true = np.array([-2.8, 4.2, -11.76])
        self.assertTrue(np.allclose(r,r_true), msg="Friction cone constraint incorrect") 

    def test_equal_timestep_constraint(self):
        """Check that the add_equal_time_constraints method executes"""
        pre_cstr = len(self.opt.prog.GetAllConstraints())
        self.opt.add_equal_time_constraints()
        post_cstr2 = len(self.opt.prog.GetAllConstraints())
        self.assertNotEqual(pre_cstr, post_cstr2, msg="Equal time constraints not added")

    def test_add_running_cost(self):
        """Check that add_running_cost executes without error"""
        Q = 10*np.ones((1,1))
        b = np.zeros((1,1))
        cost = lambda u: (u - b).dot(Q).dot(u-b)
        pre_costs = len(self.opt.prog.GetAllCosts())
        self.opt.add_running_cost(cost, vars=[self.opt.u])
        post_costs = len(self.opt.prog.GetAllCosts())
        self.assertNotEqual(pre_costs, post_costs, msg="Running cost not added")

    def test_add_quadratic_cost(self):
        """Check that add_quadratic_running_cost executes without error"""
        Q = 10*np.ones((1,1))
        b = np.zeros((1,))
        pre_costs = len(self.opt.prog.GetAllCosts())
        self.opt.add_quadratic_running_cost(Q, b, [self.opt.u])
        post_costs = len(self.opt.prog.GetAllCosts())
        self.assertNotEqual(pre_costs, post_costs, msg="Did not add quadratic cost")
        
    def test_add_final_cost(self):
        """Check that add_final_cost executes without error"""
        cost = lambda h: np.sum(h)
        pre_costs = len(self.opt.prog.GetAllCosts())
        self.opt.add_final_cost(cost, vars=[self.opt.h])
        post_costs = len(self.opt.prog.GetAllCosts())
        self.assertNotEqual(pre_costs, post_costs, msg="Final Cost not added")

    def test_add_state_constraint(self):
        """Check that add_state_constraint executes without error"""
        q0 = np.array([1,2])
        index = [0,1]
        pre_cstr = len(self.opt.prog.GetAllConstraints())
        self.opt.add_state_constraint(knotpoint=0, value=q0, subset_index=index)
        post_cstr = len(self.opt.prog.GetAllConstraints())
        self.assertNotEqual(pre_cstr, post_cstr, msg="State constraint not added")

    def test_set_initial_guess(self):
        """Check that set_initial_guess executes without error"""
        guess = np.zeros((self.opt.x.shape))
        self.opt.set_initial_guess(xtraj=guess)
        set_guess = self.opt.prog.GetInitialGuess(self.opt.x)
        self.assertTrue(np.array_equal(guess, set_guess), msg="Initial guess not set")
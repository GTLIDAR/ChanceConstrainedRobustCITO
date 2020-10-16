"""
unittests for contactimplicit.py

Luke Drnach
October 14, 2020
"""

import numpy as np
import unittest
from trajopt.contactimplicit import ContactImplicitDirectTranscription
from systems.timestepping import TimeSteppingMultibodyPlant

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
        self.h = np.ones((self.opt.h.shape[0],))
        self.u = np.ones((self.opt.u.shape[0],))
        self.x = np.zeros((self.opt.x.shape[0],))
        self.l = np.zeros((self.opt.l.shape[0],))
        context = self.model.multibody.CreateDefaultContext()
        Jn, Jt = self.model.GetContactJacobians(context)
        self.numN = Jn.shape[0]
        self.numT = Jt.shape[0]

    def test_opt_creation(self):
        """Check that the optimization can be set up"""        
        # First check that the object is not none
        self.assertIsNotNone(self.opt, msg="Optimization creation returned None")
        
    def test_eval_dynamic_constraint(self):
        """Check that the dynamic constraint can be evaluated"""
        z = np.concatenate([self.h, self.x, self.x, self.u, self.l], axis=0)
        r = self.opt._ContactImplicitDirectTranscription__backward_dynamics(z)
        self.assertEqual(r.shape[0], self.x.shape[0], msg="backwards_dynamics returns a vector with the wrong size")
        
    def test_eval_normaldist_constraint(self):
        """Check that the normal distance constraint can be evaluated"""
        z = np.concatenate([self.x, self.l[0:self.numN]], axis=0)
        r = self.opt._ContactImplicitDirectTranscription__normal_distance_constraint(z)
        self.assertEqual(r.shape[0], 3*self.numN, msg="normal distance constraint returns the wrong number of constraints")

    def test_eval_slidingvel_constraint(self):
        """Check that the sliding velocity constraint can be evaluated"""
        z = np.concatenate([self.x,self.l[self.numN:]], axis=0)
        r = self.opt._ContactImplicitDirectTranscription__sliding_velocity_constraint(z)
        self.assertEqual(r.shape[0], 3*self.numT, msg="Sliding velocity constraint returns the wrong number of constraints")

    def test_eval_friccone_constraint(self):
        """Check that the friction cone constraint can be evaluated"""
        z = np.concatenate([self.x, self.l], axis=0)
        r = self.opt._ContactImplicitDirectTranscription__friction_cone_constraint(z)
        self.assertEqual(r.shape[0], 3*self.numN, msg="Friction cone constraint returns the wrong number of constraints") 

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
        q0 = np.array([1,2,3])
        index = [0,1,2]
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
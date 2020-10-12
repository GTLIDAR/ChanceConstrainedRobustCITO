"""
test_time_stepping.py: unittests for checking the implementation in TimeSteppingMultibodyPlant.py
Luke Drnach
October 12, 2020
"""
import numpy as np
import unittest
from systems.timestepping import TimeSteppingMultibodyPlant

class TestTimeStepping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._model = TimeSteppingMultibodyPlant(file="../urdf/fallingBox.urdf")
        cls._model.Finalize()  

    def test_finalized(self):
        """Assert that the MultibodyPlant has been finalized"""
        self.assertTrue(self._model.is_finalize(), msg='MultibodyPlant not finalized')
        self.assertIsNotNone(self._model.collision_poses, msg='Finalize failed to set collision geometries')

    def test_context(self):
        """Test that MultibodyPlant can still create Contexts"""
        # Create the context
        context = self._model.CreateDefaultContext()
        self.assertIsNotNone(context, msg="Context is None")

    def test_set_positions(self):
        """Test that we can still set positions in MultibodyPlant"""
        q = [0,0,3,0,0,0]
        # Set the positions
        context = self._model.CreateDefaultContext()
        self._model.SetPositions(context, q)
        # Now get the position and check it
        self.assertListEqual(self.GetPositions(context).tolist(), q, msg="Position not set")

    def test_normal_distances(self):
        """Test that the normal distances can be calculated"""
        # Set the state above the terrain
        context = self._model.CreateDefaultContext()
        self._model.SetPositions(context, [0,0,3,0,0,0])
        # Assert that there are 8 distances
        distances = self._model.GetNormalDistances(context)
        self.assertEqual(distances.shape,(8,),msg="Contact distances are the wrong shape")
        # Check the values of the distances
        distances = np.sort(distances, axis=None)
        true_dist = np.array([2.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5])
        self.assertAlmostEqual(distances,true_dist, msg="Incorrect values for normal distances")

    def test_contact_jacobians(self):
        """Test the contact jacobians can be calculated"""
        # Set the state above the terrain
        context = self._model.CreateDefaultContext()
        self._model.SetPositions(context, [0,0,3,0,0,0])
        # Assert that there are 8 normal jacobians, and 32 tangent jacobians
        Jn, Jt = self._model.GetContactJacobians(context)
        self.assertTupleEqual(Jn.shape, (8,6), msg="Normal Jacobian has the wrong shape")
        self.assertTupleEqual(Jt.shape, (32,6), msg="Tangential Jacobian has the wrong shape")

    def test_friction_coefficients(self):
        """Test that the friction coefficients can be calculated"""
        # Set the state to above the terrain
        context = self._model.CreateDefaultContext()
        self._model.SetPositions(context,[0,0,3,0,0,0])
        # Get friction coefficients
        friction_coeff = self._model.GetFrictionCoefficients(context)
        # Check that there are 8 of them
        self.assertEqual(len(friction_coeff), 8, msg="wrong number of friction coefficients")

if __name__ == "__main__":
    unittest.main()
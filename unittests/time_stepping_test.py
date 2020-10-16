"""
test_time_stepping.py: unittests for checking the implementation in TimeSteppingMultibodyPlant.py
Luke Drnach
October 12, 2020
"""
import numpy as np
import unittest
from systems.timestepping import TimeSteppingMultibodyPlant
#TODO: Write tests for testing autodiff functions
class TestTimeStepping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._model = TimeSteppingMultibodyPlant(file="systems/urdf/fallingBox.urdf")
        cls._model.Finalize()  

    def setUp(self):
        """Set the context variable for each test"""
        self.context = self._model.multibody.CreateDefaultContext()
        self._model.multibody.SetPositions(self.context, [0,0,3,0,0,0])

    def test_finalized(self):
        """Assert that the MultibodyPlant has been finalized"""
        self.assertTrue(self._model.multibody.is_finalized(), msg='MultibodyPlant not finalized')
        self.assertIsNotNone(self._model.collision_poses, msg='Finalize failed to set collision geometries')

    def test_context(self):
        """Test that MultibodyPlant can still create Contexts"""
        # Create the context
        context = self._model.multibody.CreateDefaultContext()
        self.assertIsNotNone(context, msg="Context is None")

    def test_set_positions(self):
        """Test that we can still set positions in MultibodyPlant"""
        q = [0,0,3,0,0,0]
        # Now get the position and check it
        self.assertListEqual(self._model.multibody.GetPositions(self.context).tolist(), q, msg="Position not set")

    def test_normal_distances(self):
        """Test that the normal distances can be calculated"""
        # Assert that there are 8 distances
        distances = self._model.GetNormalDistances(self.context)
        self.assertEqual(distances.shape,(self._model.multibody.num_collision_geometries(),),msg="Contact distances are the wrong shape")
        # Check the values of the distances
        distances = np.sort(distances, axis=None)
        true_dist = np.array([2.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5])
        self.assertTrue(np.allclose(distances,true_dist), msg="Incorrect values for normal distances")

    def test_contact_jacobians(self):
        """Test the contact jacobians can be calculated"""
        # Assert that there are 8 normal jacobians, and 32 tangent jacobians
        Jn, Jt = self._model.GetContactJacobians(self.context)
        numN = self._model.multibody.num_collision_geometries()
        numT = 4*(self._model._dlevel+1)*numN
        numQ = self._model.multibody.num_positions()
        self.assertTupleEqual(Jn.shape, (numN,numQ), msg="Normal Jacobian has the wrong shape")
        self.assertTupleEqual(Jt.shape, (numT,numQ), msg="Tangential Jacobian has the wrong shape")

    def test_friction_coefficients(self):
        """Test that the friction coefficients can be calculated"""
        # Get friction coefficients
        friction_coeff = self._model.GetFrictionCoefficients(self.context)
        # Check that there are 8 of them
        self.assertEqual(len(friction_coeff), 8, msg="wrong number of friction coefficients")

    def test_autodiff_finalized(self):
        """Test that the autodiff copy is finalized"""
        # Get the autodiff copy
        copy_ad = self._model.toAutoDiffXd()
        # Check that the copy is finalized
        self.assertTrue(copy_ad.multibody.is_finalized(), msg="AutoDiff copy not finalized")

    def test_autodiff_collisions(self):
        """Test that the autodiff copy gets the collision frames"""
        # Get the autodiff copy
        copy_ad = self._model.toAutoDiffXd()
        # Check that the number of collision frames and poses is equal to the original
        self.assertEqual(len(copy_ad.collision_frames), len(self._model.collision_frames), msg = "wrong number of collision frames")
        self.assertEqual(len(copy_ad.collision_poses), len(self._model.collision_poses), msg = "wrong number of collision poses")

if __name__ == "__main__":
    unittest.main()
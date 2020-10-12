"""
TimeSteppingMultibodyPlant: an extension of pyDrake's MultibodyPlant for use with ContactImplicitDirectTranscription

TimeSteppingMultibodyPlant extends pyDrake's MultibodyPlant class and adds a few helper methods for use with ContactImplicitDirectTranscription. In building the model, TimeSteppingMultibodyPlant also constructs a SceneGraph and stores the relevant information about the contact geometry stored in the model. TimeSteppingMultibodyPlant also supports arbitrary terrain geometries by adding a custom terrain object to the class. The terrain specification is used to calculate normal distances and contact Jacobians in TimeSteppingMultibodyPlant

This class is not fully integrated into pyDrake. For example, this class uses a custom implementation of the terrain instead of using a terrain model from pyDrake, and likewise uses custom methods to calculate collision distances. In future, the class may be removed and the operations replaced by pure pyDrake operations.

Luke Drnach
October 9, 2020
"""
import numpy as np
from math import pi
from pydrake.all import MultibodyPlant, DiagramBuilder, SceneGraph,AddMultibodyPlantSceneGraph, JacobianWrtVariable, AngleAxis, RotationMatrix
from pydrake.multibody.parsing import Parser
from TimeStepping.Terrain import FlatTerrain
from utilities import FindResource

class TimeSteppingMultibodyPlant(MultibodyPlant):
    """
    """
    def __init__(self, file=None, terrain=FlatTerrain, dlevel=0):
        """
        Initialize TimeSteppingMultibodyPlant with a model from a file and an arbitrary terrain geometry
        """
        # Initialize the MultibodyPlant first
        super().__init__(0.0)
        # Store the terrain
        self.terrain = terrain
        self._dlevel=0
        # Build the MultibodyPlant from the file, if one exists
        self.model_index = []
        if file is not None:
            # Parse the file
            self.model_index = Parser(self).AddModelFromFile(FindResource(file))
        # Initialize the collision data
        self.collision_frames = []
        self.collision_poses = []

    def Finalize(self):
        """
        Cements the topology of the MultibodyPlant and identifies all available collision geometries. 
        """
        # Setup the diagram
        builder = DiagramBuilder()
        scene_graph = SceneGraph()
        AddMultibodyPlantSceneGraph(builder, self, scene_graph)
        inspector = scene_graph.model_inspector()
        # Finalize the model so we can access the collision geometry
        super().Finalize()
        # Locate collision geometries and contact points
        body_inds = self.GetBodyIndices(self.model_index)
        # Get the collision frames for each body in the model
        for body_ind in body_inds:
            body = self.get_body(body_ind)
            collision_ids = self.GetCollisionGeometriesForBody(body)
            for id in collision_ids:
                # get and store the collision geometry frames
                self.collision_frames.append(inspector.GetFrameId(id))
                self.collision_poses.append(inspector.GetPoseInFrame(id))

    def GetNormalDistances(self, context):   
        """
        Returns an array of signed distances between the contact geometries and the terrain, given the current system context

        Arguments:
            context: a pyDrake MultibodyPlant context
        Return values:
            distances: a numpy array of signed distance values
        """
        distances = []
        for frame, pose in zip(self.collision_frames, self.collision_poses):
            # Transform collision frames to world coordinates
            collision_pt = self.CalcPointsPositions(context, frame, pose.translation(), self.world_frame()) 
            # Calc nearest point on terrain in world coordinates
            terrain_pt = self.terrain.nearest_point(collision_pt)
            # Calc normal distance to terrain   
            terrain_frame = self.terrain.local_frame(terrain_pt)  
            normal = terrain_frame[0,:]
            distances.append(normal.dot(collision_pt - terrain_pt))
        # Return the distances as a single array
        return np.concatentate(distances, axis=0)

    def GetContactJacobians(self, context):
        """
        Returns a tuple of numpy arrays representing the normal and tangential contact Jacobians evaluated at each contact point

        Arguments:
            context: a pyDrake MultibodyPlant context
        Return Values
            (Jn, Jt): the tuple of contact Jacobians. Jn represents the normal components and Jt the tangential components
        """
        Jn = []
        Jt = []
        for frame, pose in zip(self.collision_frames, self.collision_poses):
            # Transform collision frames to world coordinates
            collision_pt = self.CalcPointsPositions(context, frame, pose.translation(), self.world_frame())
            # Calc nearest point on terrain in world coordinates
            terrain_pt = self.terrain.nearest_point(collision_pt)
            # Calc normal distance to terrain   
            terrain_frame = self.terrain.local_frame(terrain_pt)
            normal, tangent = np.split(terrain_frame, [1], axis=0)
            # Discretize to the chosen level 
            tangent = self._discretize_friction(normal, tangent)  
            # Get the contact point Jacobian
            J = self.CalcJacobianTranslationalVelocity(context,
                 JacobianWrtVariable.kQDot,
                 frame,
                 pose.translation(),
                 self.world_frame(),
                 self.world_frame())
            # Calc contact Jacobians
            Jn.append(normal.transpose().dot(J))
            Jt.append(tangent.transpose().dot(J))
        # Return the Jacobians as a tuple of np arrays
        return (np.concatenate(Jn, axis=0), np.concatenate(Jt, axis=0))    

    def GetFrictionCoefficients(self, context):
        """
        Return friction coefficients for nearest point on terrain
        
        Arguments:
            context: the current MultibodyPlant context
        Return Values:
            friction_coeff: list of friction coefficients
        """
        friction_coeff = []
        for frame, pose in zip(self.collision_frames, self.collision_poses):
            # Transform collision frames to world coordinates
            collision_pt = self.CalcPointsPositions(context, frame, pose.translation(), self.world_frame())
            # Calc nearest point on terrain in world coordiantes
            terrain_pt = self.terrain.nearest_point(collision_pt)
            friction_coeff.append(self.terrain.get_friction(terrain_pt))
        # Return list of friction coefficients
        return friction_coeff

    def set_discretization_level(self, dlevel=0):
        """Set the friction discretization level. The default is 0"""
        self._dlevel = dlevel

    def _discretize_friction(self, normal, tangent):
        """
        Rotates the terrain tangent vectors to discretize the friction cone
        
        Arguments:
            normal:  The terrain normal direction, (1x3) numpy array
            tangent:  The terrain tangent directions, (2x3) numpy array
        Return Values:
            all_tangents: The discretized friction vectors, (2nx3) numpy array
        """
        # Reflect the current friction basis
        tangent = np.concatenate((tangent, -tangent), axis=0)
        all_tangents = tangent
        # Rotate the tangent basis around the normal vector
        for n in range(1, self._dlevel+1):
            # Create an angle-axis representation of rotation
            R = RotationMatrix(theta_lambda=AngleAxis(angle=n*pi/(2*(self._dlevel+1)), axis=normal))
            # Apply the rotation matrix
            all_tangents.append(R.multiply(tangent.transpose()).transpose())
        return np.concatenate(all_tangents, axis=0)

"""
TimeSteppingMultibodyPlant: an container for pyDrake's MultibodyPlant for use with ContactImplicitDirectTranscription

TimeSteppingMultibodyPlant instantiates pyDrake's MultibodyPlant class and adds a few helper methods for use with ContactImplicitDirectTranscription. In building the model, TimeSteppingMultibodyPlant also constructs a SceneGraph and stores the relevant information about the contact geometry stored in the model. TimeSteppingMultibodyPlant also supports arbitrary terrain geometries by adding a custom terrain object to the class. The terrain specification is used to calculate normal distances and contact Jacobians in TimeSteppingMultibodyPlant

This class is not fully integrated into pyDrake. For example, this class uses a custom implementation of the terrain instead of using a terrain model from pyDrake, and likewise uses custom methods to calculate collision distances. In future, the class may be removed and the operations replaced by pure pyDrake operations.

Note that, due to issues with pybind, TimeSteppingMultibodyPlant does NOT subclass MultibodyPlant. Instead, TimeSteppingMultibodyPlant instantiates MultibodyPlant as property called plant.

Luke Drnach
October 9, 2020
"""
import numpy as np
from math import pi
from pydrake.all import MultibodyPlant, DiagramBuilder, SceneGraph,AddMultibodyPlantSceneGraph, JacobianWrtVariable, AngleAxis, RotationMatrix, RigidTransform
from pydrake.multibody.parsing import Parser
from systems.terrain import FlatTerrain
from utilities import FindResource
#TODO: Implemet toAutoDiffXd method to convert to autodiff class


class TimeSteppingMultibodyPlant():
    """
    """
    def __init__(self, file=None, terrain=FlatTerrain(), dlevel=0):
        """
        Initialize TimeSteppingMultibodyPlant with a model from a file and an arbitrary terrain geometry. Initialization also welds the first frame in the MultibodyPlant to the world frame
        """
        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, 0.001)
        # Store the terrain
        self.terrain = terrain
        self._dlevel=0
        # Build the MultibodyPlant from the file, if one exists
        self.model_index = []
        if file is not None:
            # Parse the file
            self.model_index = Parser(self.plant).AddModelFromFile(FindResource(file))
            # Weld the first frame to the world-frame
            body_inds = self.plant.GetBodyIndices(self.model_index)
            base_frame = self.plant.get_body(body_inds[0]).body_frame()
            self.plant.WeldFrames(self.plant.world_frame(), base_frame, RigidTransform())
        # Initialize the collision data
        self.collision_frames = []
        self.collision_poses = []

    def Finalize(self):
        """
        Cements the topology of the MultibodyPlant and identifies all available collision geometries. 
        """
        # Finalize the underlying plant model
        self.plant.Finalize()
        # Idenify and store collision geometries
        self.__store_collision_geometries()

    def GetNormalDistances(self, context):   
        """
        Returns an array of signed distances between the contact geometries and the terrain, given the current system context

        Arguments:
            context: a pyDrake MultibodyPlant context
        Return values:
            distances: a numpy array of signed distance values
        """
        qtype = self.plant.GetPositions(context).dtype
        nCollisions = len(self.collision_frames)
        distances = np.zeros((nCollisions,), dtype=qtype)
        for n in range(0, nCollisions):
            # Transform collision frames to world coordinates
            collision_pt = self.plant.CalcPointsPositions(context, 
                                        self.collision_frames[n],
                                        self.collision_poses[n].translation(),
                                        self.plant.world_frame()) 
            # Calc nearest point on terrain in world coordinates
            terrain_pt = self.terrain.nearest_point(collision_pt)
            # Calc normal distance to terrain   
            terrain_frame = self.terrain.local_frame(terrain_pt)  
            normal = terrain_frame[0,:]
            distances[n] = normal.dot(collision_pt - terrain_pt)
        # Return the distances as a single array
        return distances

    def GetContactJacobians(self, context):
        """
        Returns a tuple of numpy arrays representing the normal and tangential contact Jacobians evaluated at each contact point

        Arguments:
            context: a pyDrake MultibodyPlant context
        Return Values
            (Jn, Jt): the tuple of contact Jacobians. Jn represents the normal components and Jt the tangential components
        """
        qtype = self.plant.GetPositions(context).dtype
        nCollision = len(self.collision_frames)
        Jn = np.zeros((nCollision, self.plant.num_positions()),dtype=qtype)
        Jt = np.zeros((nCollision * 4 * (self._dlevel+1), self.plant.num_positions()),dtype=qtype)
        for n in range(0, nCollision):
            # Transform collision frames to world coordinates
            collision_pt = self.plant.CalcPointsPositions(context,
                                        self.collision_frames[n],
                                        self.collision_poses[n].translation(),
                                        self.plant.world_frame())
            # Calc nearest point on terrain in world coordinates
            terrain_pt = self.terrain.nearest_point(collision_pt)
            # Calc normal distance to terrain   
            terrain_frame = self.terrain.local_frame(terrain_pt)
            normal, tangent = np.split(terrain_frame, [1], axis=0)
            # Discretize to the chosen level 
            tangent = self.__discretize_friction(normal, tangent)  
            # Get the contact point Jacobian
            J = self.plant.CalcJacobianTranslationalVelocity(context,
                 JacobianWrtVariable.kQDot,
                 self.collision_frames[n],
                 self.collision_poses[n].translation(),
                 self.plant.world_frame(),
                 self.plant.world_frame())
            # Calc contact Jacobians
            Jn[n,:] = normal.dot(J)
            Jt[n*4*(self._dlevel+1) : (n+1)*4*(self._dlevel+1), :] = tangent.dot(J)
        # Return the Jacobians as a tuple of np arrays
        return (Jn, Jt)    

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
            collision_pt = self.plant.CalcPointsPositions(context, frame, pose.translation(), self.plant.world_frame())
            # Calc nearest point on terrain in world coordiantes
            terrain_pt = self.terrain.nearest_point(collision_pt)
            friction_coeff.append(self.terrain.get_friction(terrain_pt))
        # Return list of friction coefficients
        return friction_coeff

    def toAutoDiffXd(self):
        """Covert the MultibodyPlant to use AutoDiffXd instead of Float"""

        # Create a new TimeSteppingMultibodyPlant model
        copy_ad = TimeSteppingMultibodyPlant(file=None, terrain=self.terrain, dlevel=self._dlevel)
        # Instantiate the plant as the Autodiff version
        copy_ad.plant = self.plant.ToAutoDiffXd()
        copy_ad.scene_graph = self.scene_graph.ToAutoDiffXd()
        copy_ad.model_index = self.model_index
        # Store the collision frames to finalize the model
        copy_ad.__store_collision_geometries()
        return copy_ad

    def set_discretization_level(self, dlevel=0):
        """Set the friction discretization level. The default is 0"""
        self._dlevel = dlevel

    def __store_collision_geometries(self):
        """Identifies the collision geometries in the model and stores their parent frame and pose in parent frame in lists"""
        # Create a diagram and a scene graph
        inspector = self.scene_graph.model_inspector()
        # Locate collision geometries and contact points
        body_inds = self.plant.GetBodyIndices(self.model_index)
        # Get the collision frames for each body in the model
        for body_ind in body_inds:
            body = self.plant.get_body(body_ind)
            collision_ids = self.plant.GetCollisionGeometriesForBody(body)
            for id in collision_ids:
                # get and store the collision geometry frames
                frame_name = inspector.GetName(inspector.GetFrameId(id)).split("::")
                self.collision_frames.append(self.plant.GetFrameByName(frame_name[-1]))
                self.collision_poses.append(inspector.GetPoseInFrame(id))

    def __discretize_friction(self, normal, tangent):
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
        all_tangents = np.zeros((4*(self._dlevel+1), tangent.shape[1]))
        all_tangents[0:4, :] = tangent
        # Rotate the tangent basis around the normal vector
        for n in range(1, self._dlevel+1):
            # Create an angle-axis representation of rotation
            R = RotationMatrix(theta_lambda=AngleAxis(angle=n*pi/(2*(self._dlevel+1)), axis=normal))
            # Apply the rotation matrix
            all_tangents[n*4 : (n+1)*4, :] = R.multiply(tangent.transpose()).transpose()
        return all_tangents

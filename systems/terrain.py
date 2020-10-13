"""
terrain.py: package for specifying arbitrary terrain geometries for use with TimeSteppingMultibodyPlant.py
Luke Drnach
October 12, 2020
"""
import numpy as np
from abc import ABC, abstractmethod 

class Terrain(ABC):
    """
    Abstract class outlining methods required for specifying a terrain geometry that can be used with TimeSteppingMultibodyPlant
    """
    @abstractmethod
    def nearest_point(self, x):
        """
        Returns the nearest point on the terrain to the supplied point x

        Arguments:
            x: the point of interest
        Return values:
            y: the point on the terrain which is nearest to x
        """

    @abstractmethod 
    def local_frame(self, x):
        """
        Returns the local coordinate frame of the terrain at the supplied point x

        Arguments:
            x: a point on the terrain
        Return values:
            R: an array. The first row is the terrain normal vector, the remaining rows are the terrain tangential vectors
        """

    @abstractmethod
    def get_friction(self, x):
        """
        Returns the value of terrain friction coefficient at the supplied point

        Arguments:
            x: a point on the terrain
        Return values
            fric_coeff: a scalar friction coefficients
        """

class FlatTerrain2D(Terrain):
    """
    Implementation of a 2-dimensional terrain with flat geometry
    """
    def __init__(self, height = 0, friction = 0.5):
        """ Construct the terrain, set it's height and friction coefficient"""
        self.height = height
        self.friction  = friction

    def nearest_point(self, x):
        """
        Returns the nearest point on the terrain to the supplied point x

        Arguments:
            x: (2x1), the point of interest
        Return values:
            y: (2x1), the point on the terrain which is nearest to x
        """
        return np.array([x[0], self.height])

    def local_frame(self, _):
        """
        Returns the local coordinate frame of the terrain at the supplied point x

        Arguments:
            x: (2x1) a point on the terrain
        Return values:
            R: (2x2), an array. The first row is the terrain normal vector, the remaining rows are the terrain tangential vectors
        """
        return np.array([[0,1],[1,0]])
    
    def get_friction(self, _):
        """
        Returns the value of terrain friction coefficient at the supplied point

        Arguments:
            x: (2x1) a point on the terrain
        Return values
            fric_coeff: a scalar friction coefficients
        """
        return self.friction

class FlatTerrain(Terrain):
    """ Implementation of 3-dimensional flat terrain with no slope """
    def __init__(self, height=0.0, friction=0.5):
        """ Constructs the terrain with the specified height and friction coefficient """
        self.height = height
        self.friction = friction
    
    def nearest_point(self, x):
        """
        Returns the nearest point on the terrain to the supplied point x

        Arguments:
            x: (3x1), the point of interest
        Return values:
            y: (3x1), the point on the terrain which is nearest to x
        """
        terrain_pt = np.copy(x)
        terrain_pt[-1] = self.height
        return terrain_pt

    def local_frame(self, _):
        """
        Returns the local coordinate frame of the terrain at the supplied point x

        Arguments:
            x: (3x1), a point on the terrain
        Return values:
            R: a (3x3) array. The first row is the terrain normal vector, the remaining rows are the terrain tangential vectors
        """
        return np.array([[0.,0.,1.], [1.,0.,0.], [0.,1.,0.]])

    def get_friction(self, _):
        """
        Returns the value of terrain friction coefficient at the supplied point

        Arguments:
            x: (3x1) a point on the terrain
        Return values
            fric_coeff: a scalar friction coefficients
        """
        return self.friction

if __name__ == "__main__":
    print("Hello world")
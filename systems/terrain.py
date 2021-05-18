"""
terrain.py: package for specifying arbitrary terrain geometries for use with TimeSteppingMultibodyPlant.py
Luke Drnach
October 12, 2020
"""
import numpy as np
from abc import ABC, abstractmethod 
from functools import partial

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

class StepTerrain(FlatTerrain):
    """ Implementation of 3-dimensional terrain with a step in the 1st dimension"""
    def __init__(self, height = 0.0, step_height=1.0, step_location=1.0, friction=0.5):
        """ Construct the terrain and set the step x-location and step height"""
        super().__init__(height, friction)
        self.step_height = step_height
        self.step_location = step_location

    def nearest_point(self, x):
        """
        Returns the nearest point on the terrain to the supplied point x

        Arguments:
            x: (3x1), the point of interest
        Return values:
            y: (3x1), the point on the terrain which is nearest to x
        """
        terrain_pt = np.copy(x)
        if x[0] < self.step_location:
            terrain_pt[-1] = self.height
        else:
            terrain_pt[-1] = self.step_height 
        return terrain_pt           

class SlopeStepTerrain(FlatTerrain):
    """ Implementation of a piecewise linear terrain with a slope"""
    def __init__(self, height, slope, slope_location, friction):
        """ Construct the terrain with the specified slope, starting at x = slope_location """
        super().__init__(height, friction)
        self.slope = slope
        self.slope_location = slope_location

    def nearest_point(self, x):
        """
        Returns the nearest point on the terrain to the supplied point x

        Arguments:
            x: (3x1), the point of interest
        Return values:
            y: (3x1), the point on the terrain which is nearest to x
        """
        terrain_pt = np.copy(x)
        if self._on_flat(x):
            terrain_pt[-1] = self.height
        else:
            terrain_pt[-1] = self.slope * (terrain_pt[0] - self.slope_location) + self.height
        return terrain_pt

    def local_frame(self, x):
        """
        Returns the local coordinate frame of the terrain at the supplied point x

        Arguments:
            x: (3x1), a point on the terrain
        Return values:
            R: a (3x3) array. The first row is the terrain normal vector, the remaining rows are the terrain tangential vectors
        """
        if self._on_flat(x):
            return np.array([[0., 0., 1.],[1., 0., 0.], [0., 1., 0.]])
        else:
            B = np.array([[-self.slope, 0., 1.],[1., 0., self.slope],[0., 1., 0.]])
            return B/np.sqrt(1 + self.slope **2)

    def _on_flat(self, x):
        """ Check if the point is on the flat part of the terrain or not """
        return x[-1] < self.height - self.slope *(x[0] - self.slope_location)

class VariableFrictionFlatTerrain(FlatTerrain):
    def __init__(self, height=0.0, fric_func=None):
        """ Constructs a flat terrain with variable friction """
        super.__init__(height, friction=0.0)
        if fric_func is None:
            self.friction_function = partial(constant_friction, c=0.5)
        else:
            self.friction_function = fric_func

    def get_friction(self, x):
        """
        Returns the value of terrain friction coefficient at the supplied point

        Arguments:
            x: (3x1) a point on the terrain
        Return values
            fric_coeff: a scalar friction coefficients
        """
        return self.friction_function(x)

def constant_friction(_, c):
    """ Parameterized constant friction function """
    return c

if __name__ == "__main__":
    print("Hello world")
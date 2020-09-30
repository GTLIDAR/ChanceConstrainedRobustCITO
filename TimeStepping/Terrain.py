import numpy as np
from abc import ABC, abstractmethod 

class Terrain(ABC):
    @abstractmethod
    def nearest(self,x):
        pass
    def basis(self, x):
        pass

class FlatTerrain2D(Terrain):
    def __init__(self, height = 0, friction = 0.5):
        self.height = height
        self.friction  = friction

    def nearest(self, x):
        return np.array([x[0], self.height])

    def basis(self, x):
        return (np.array([0,1]), np.array([1,0]))

if __name__ == "__main__":
    print("Hello world")
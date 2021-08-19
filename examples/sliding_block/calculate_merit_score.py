import numpy as np
'''
This script calculates merit score for friction and normal distance constraint violations
'''

def normal_distance_merit_function(F, phi):
    '''
    This method calculates the merit score as follows:
    M(z;p) = G_ec(z;p)^2 + min (0, G_ic(z;p)^2)

    Inputs: F is the normal force 1xN matrix
            phi is the normal distance 1xN matrix
    
    Outputs: M is the merit score scalar
    '''
    
    M_eq = np.sum(np.square(np.multiply(F, phi)))
    M_iq = np.sum(np.square(np.minimum(0, phi))) + np.sum(np.square(np.minimum(0, F)))
    M = M_eq + M_iq

    return M

def friction_cone_merit_function():

    pass

if __name__ == "__main__":
    F = np.array([1,1,1])
    phi = np.array([2,1,-1])
    M = normal_distance_merit_function(F, phi)
    friction = np.loadtxt('data/slidingblock/erm/friction.txt')
    num = 0.1
    text = "erm_%d" %(num)
    print(text)

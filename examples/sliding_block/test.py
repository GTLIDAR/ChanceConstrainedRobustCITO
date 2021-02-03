import numpy as np
from scipy.stats import norm
from scipy.stats import rv_continuous as rv
from scipy.stats import rv_discrete as ds
from math import *
class ParentClass: 
    def __init__(self):
        self.f()
        self.h()
    def f(self): 
        print("Hi!"); 
    def h(self):
        print("bye")

class ChildClass(ParentClass): 
    def __init__(self):
        super(ChildClass, self).__init__()
        # print(np.zeros(5,))
        
        # self.f()
        # self.h()
    def f(self):
        
        # print(super(ChildClass, self))
        # # print(super())
        # self.h()
        print("Hello!")
    def _pdf (self, x, mean, sd):
        prob_density = (1/(2*np.pi*sd**2) ) * np.exp(-0.5*((x-mean)/sd)**2)
        return prob_density
    # def h(self):
    #     super(ChildClass, self).h()
    #     print("bye bye")

# A = ChildClass()
# A.f()
# A.h()
def _pdf ( x, mean, sd):
    prob_density = (1/(sd * np.sqrt(2 * np.pi)) ) * np.exp(-0.5*((x-mean)**2/sd**2))
    return prob_density

def _cdf ( x, mean, sd):
    cum_dist = np.zeros(len(x))
    # for i in range(len(x)):
    #     A = _erf((x[i] - mean)/(sd * np.sqrt(2)))
    #     cum_dist[i] = 1/2 *(1 + A)
    A = _erf((x - mean)/(sd * np.sqrt(2)))
    cum_dist = 1/2 *(1 + A)
    return cum_dist
        
def _erf( x):
# save the sign of x
    # sign = 1 if x >= 0 else -1
    sign = np.zeros(len(x))
    sign[x >= 0] = 1
    sign[x < 0] = -1 
    x = abs(x)
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign*y

A = np.array([-2, -1, 0, 1, 2])
B = np.ones(5,) * 5
mu = 2
sigma  = 1
norm_A = _cdf(A, mu, sigma)
norm_B = norm.cdf(A, mu, sigma)
# normB = rv.pdf(A, mu, sigma)
print(norm_A)
print(norm_B)
# print(np.array([A]))
# print(np.array(A.shape))
# A = np.sum(A)
# print(A)
# print(type(A))
# print(A.shape)
# A = np.array(A)
# print(A.shape)

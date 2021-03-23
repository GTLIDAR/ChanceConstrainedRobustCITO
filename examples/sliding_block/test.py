import numpy as np
from scipy.stats import norm
from scipy.stats import rv_continuous as rv
from scipy.stats import rv_discrete as ds
from math import *
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
    sign = np.zeros(x.shape)
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
def ermCost(x, mu, sigma):
    """
    Gaussian ERM implementation
    """
    # initialize pdf and cdf
    pdf = norm.pdf(x, mu, sigma)
    cdf = norm.cdf(x, mu, sigma)
    f = x**2 - sigma**2 * (x + mu) * pdf + (sigma**2+ mu**2 - x**2) * cdf
    return f

def ermCost_test(x, mu, sigma):
    pdf = _pdf(x, mu, sigma)
    cdf = _cdf(x, mu, sigma)
    f = x**2 - sigma**2 * (x + mu) * pdf + (sigma**2+ mu**2 - x**2) * cdf
    return f
x = np.array([1,13])
mu = np.array([2,2])
sigma = 1
cost = ermCost(x, mu, sigma)
cost_test = ermCost_test(x, mu, sigma)

print(cost)
print(cost_test)


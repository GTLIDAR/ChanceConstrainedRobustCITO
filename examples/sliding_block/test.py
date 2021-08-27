import numpy as np
from scipy.stats import norm
from scipy.stats import rv_continuous as rv
from scipy.stats import rv_discrete as ds
from math import *
import matplotlib.pyplot as plt
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

def error_plot_test():
    fig = plt.figure()
    x = 1
    y = 2.5 * np.sin(x / 20 * np.pi)
    # yerr = np.linspace(0.05, 0.2, 10)
    yerr = np.zeros([2,1])
    yerr[0]= 0.1
    yerr[1]=0.2
    # yerr = np.array([0.1, 0.2])

    plt.errorbar(x, y + 3, yerr=yerr, label='both limits (default)')

    plt.errorbar(x, y + 2, yerr=yerr, uplims=True, label='uplims=True')

    plt.errorbar(x, y + 1, yerr=yerr, uplims=True, lolims=True,
                label='uplims=True, lolims=True')

    # upperlimits = [True, False] * 5
    # lowerlimits = [False, True] * 5
    # plt.errorbar(x, y, yerr=yerr, uplims=upperlimits, lolims=lowerlimits,
    #             label='subsets of uplims and lolims')

    plt.legend(loc='lower right')
    plt.show()
def barplot():
    import matplotlib.pyplot as plt 
    plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt

    objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
    y_pos = np.arange(len(objects))
    performance = [10,8,6,1e-5,2,1]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    # Get the axes object
    ax = plt.gca()
    # remove the existing ticklabels
    ax.set_xticklabels([])
    # remove the extra tick on the negative bar
    ax.set_xticks([idx for (idx, x) in enumerate(performance) if x > 0])
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # placing each of the x-axis labels individually
    label_offset = 0.5
    for language, (x_position, y_position) in zip(objects, enumerate(performance)):
        if y_position > 0:
            label_y = -label_offset
        else:
            label_y = y_position - label_offset
        ax.text(x_position, label_y, language, ha="center", va="top")
    # Placing the x-axis label, note the transformation into `Axes` co-ordinates
    # previously data co-ordinates for the x ticklabels
    ax.text(0.5, -0.05, "Usage", ha="center", va="top", transform=ax.transAxes)
    ax.set_yscale('log')
    plt.show()

if __name__ == "__main__":
    # error_plot_test()
    barplot()


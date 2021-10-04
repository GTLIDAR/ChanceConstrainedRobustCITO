"""
Useful function decorators for robotics and optimization problems

Luke Drnach
June 18, 2021
"""
import timeit
import functools
from matplotlib import pyplot as plt
import utilities as utils

def timer(func):
    """Print the runtime of the decorated function. The decorator also records the time in the total_time attribute"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = timeit.default_timer()
        value = func(*args, **kwargs)
        stop = timeit.default_timer()
        wrapper_timer.total_time = stop - start
        print(f"Finished {func.__name__!r} in {wrapper_timer.total_time:.4f} seconds")
        return value
    wrapper_timer.total_time = 0
    return wrapper_timer

def saveable_fig(func):
    """Save a figure from the output"""
    @functools.wraps(func)
    def wrapper_saveable_fig(*args, **kwargs):
        # Pull the "filename" key from kwargs, if it exists
        savename = kwargs.pop('savename', False)
        # Get the figure from the function
        fig, *axs, = func(*args, **kwargs)
        # Check kwargs for a savename and save the figure
        if savename:
            if hasattr(fig, '__iter__'):
                n = 0
                for f in fig:
                    append_str = f"_({n})"
                    f.savefig(utils.append_filename(savename, append_str), dpi=f.dpi)
                    n+=1
            else:
                fig.savefig(savename, dpi=fig.dpi)
        return (fig, *axs) 
    return wrapper_saveable_fig

def showable_fig(func):
    """Show a figure based on a show argument"""
    @functools.wraps(func)
    def wrapper_showable_fig(*args, **kwargs):
        # Check for the 'show' keyword and remove it
        show = kwargs.pop('show', False)
        # Create  the figures and axes
        out = func(*args, **kwargs)
        # Show the figure if desired
        if show:
            plt.show()
        return out
    return wrapper_showable_fig

#NOTE: TO show and save figures, add the decorator saveable_fig first and then showable_fig. This way, the figures are saved before they are shown (and ultimately deleted). 
@showable_fig
@saveable_fig
def testfig():
    fig, axs = plt.subplots(1,1)
    fig2, axs2 = plt.subplots(2,1)
    axs.plot(0, 0)
    axs2[0].plot(1,1)
    axs2[1].plot(2,2)
    return [fig, fig2], [axs, axs2]

@timer
def waste_time(num_times):
    for _ in range(num_times):
        sum([i**2 for i in range(10000)])

if __name__ == '__main__':
    fig, axs = testfig(show=True, savename='TestFigures.png')
    print(f"Got two outputs: fig = {fig} and axs = {axs}")
    print(f"Wasting time")
    waste_time(100)
    print(f"Wasted {waste_time.total_time}")

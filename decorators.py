"""
Useful function decorators for robotics and optimization problems

Luke Drnach
June 18, 2021
"""
import timeit
import functools

def timer(func):
    """Print the runtime of the decorated function. The decorator also records the time in the total_time attribute"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        wrapper_timer.total_time = 0.
        start = timeit.default_timer()
        value = func(*args, **kwargs)
        stop = timeit.default_timer()
        wrapper_timer.total_time = stop - start
        print(f"Finished {func.__name__!r} in {wrapper_timer.total_time:.4f} seconds")
        return value
    return wrapper_timer
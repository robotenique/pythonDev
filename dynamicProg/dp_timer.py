import timeit as timer

# Function argument decorator
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

# Measure the time of a func, then print it
def timer_function(func_args, name="Current function", niter = 30):
    t = timer.timeit(func_args, number=niter)
    print(f"{t} seconds ({name} , n_iter = {niter})")

def timer_wrap(func, args, niter):
    t = timer_function(wrapper(func, args), name=func.__name__, niter=niter)

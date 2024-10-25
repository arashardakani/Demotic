import time


def timer(func):
    """Time the execution of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"Function {func.__name__} took {elapsed_time:.6f} seconds to execute.")
        return elapsed_time, result

    return wrapper

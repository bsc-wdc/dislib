import numpy as np

def _sum_and_mult(arr, a=0, axis=0, b=1):
    return (np.sum(arr, axis=axis) + a) * b
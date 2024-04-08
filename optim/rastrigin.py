import numpy as np

def rastrigin(position):
    n = len(position)
    return 10*n + sum([x**2 - 10*np.cos(2*np.pi*x) for x in position])

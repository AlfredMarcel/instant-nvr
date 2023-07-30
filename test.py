import numpy as np
from itertools import product

def generate_array(x):
    combinations = list(product([0, 1], repeat=x))
    array = np.array(combinations)
    return array

x = 6
result = generate_array(x)

print(result)

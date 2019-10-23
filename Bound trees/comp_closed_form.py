import numpy as np
from scipy.special import binom


"""
@author: sdamrich
Computes the number of spanning trees in a grid-graph by the closed form formula
of theorem 1 in
Tzeng, Wu: "Spanning trees on hypercubic lattices and nonorientable surface"
"""
# dimensions of the grid
n_1 = np.float64(272)
n_2 = np.float64(87)


# computes the log10 of the product in the closed form formula
def comp_log_product(n_1, n_2):
    log_prod = np.float64(0)
    for i in range(int(n_1)):
        for j in range(int(n_2)):
            if i == 0 and j == 0: continue
            log_prod += np.log10(2- np.cos(np.pi*i/n_1)-np.cos(np.pi*j/n_2))
    return log_prod

# computes the log10 of the closed form formula
def comp_log_tree_number(n_1, n_2):
    return (n_1*n_2-1)*np.log10(2) - np.log10(n_1*n_2) + comp_log_product(n_1, n_2)



log_num_trees = comp_log_tree_number(n_1, n_2)

# num_trees = 10**log_num_trees


print("n_1: {}, n_2: {}, log_num trees: {}".format(n_1, n_2, log_num_trees))

# print("n_1: {}, n_2: {}, num trees: {}".format(n_1, n_2, num_trees))

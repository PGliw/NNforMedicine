import numpy as np

def distribution_fun(xs):
    '''
    :param xs: list of values 
    :return: list of distribution function values (for each x)
    '''
    n = len(xs)
    ys = []
    for t in range(n):
        sum = 0
        for x in range(t):
            sum += 1
        f_t = sum / n
        ys.append(f_t)
    return ys


#def kolmogorov(xs, ys):

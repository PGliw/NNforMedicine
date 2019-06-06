import csv
from utils import divide_set_to_random_parts

xs = [1, 2, 3, 4, 5]
ys = [1, 2, 3, 4, 5]

xs_s, ys_s = divide_set_to_random_parts(xs, ys)
print(xs, ys)
print(xs_s, ys_s)

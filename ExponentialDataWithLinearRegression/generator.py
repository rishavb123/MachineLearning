import numpy as np
import random


def create_dataset(num_of_datapoints, variance, step=3, correlation=''):
    val = 1
    y_arr = []
    for i in range(num_of_datapoints):
        if variance != 0:
            y_arr.append(val + random.randrange(-variance, variance))
        else:
            y_arr.append(val)

        if correlation and correlation == 'pos':
            val += step*i
        elif correlation and correlation == 'neg':
            val -= step*i
    x_arr = [i for i in range(num_of_datapoints)]
    return np.array(x_arr, dtype=np.float64), np.array(y_arr, dtype=np.float64)

xs, ys = create_dataset(60, 60, 2, correlation='pos')


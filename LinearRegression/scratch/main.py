import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def create_dataset(num_of_datapoints, variance, step=2, correlation=''):
    val = 1
    y_arr = []
    for i in range(num_of_datapoints):
        if variance != 0:
            y_arr.append(val + random.randrange(-variance, variance))
        else:
            y_arr.append(val)

        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    x_arr = [i for i in range(num_of_datapoints)]
    return np.array(x_arr, dtype=np.float64), np.array(y_arr, dtype=np.float64)


def mean(arr):
    return sum(arr)/len(arr)


def best_fit_slope(x_arr, y_arr):
    return (mean(x_arr)*mean(y_arr) - mean(x_arr*y_arr))/(mean(x_arr)**2 - mean(x_arr**2))


def best_fit_intercept(x_arr, y_arr, slope):
    return mean(y_arr) - slope*mean(x_arr)


def predict(predict_x, slope, intercept):
    return slope*predict_x + intercept


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = np.array([mean(ys_orig) for _ in ys_orig])
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_mean)


xs, ys = create_dataset(40, 10, 2, correlation='pos')

m = best_fit_slope(xs, ys)
b = best_fit_intercept(xs, ys, m)

regression_line = np.array([m*x + b for x in xs])

r_squared = coefficient_of_determination(ys, regression_line)
print r_squared

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.scatter(42, predict(42, m, b), s=100, color='g')
plt.show()

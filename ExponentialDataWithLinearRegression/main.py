import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import random


def create_dataset_2(num_of_datapoints, variance, step=3, correlation=''):
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

def makeData_2(size):
    xs, ys = create_dataset_2(size, 2*size, 2, 'neg')
    d = []
    for i in range(len(xs)):
        d.append((xs[i], ys[i]))
    return d

def create_dataset(f, num_of_datapoints, variance):
    y_arr = []
    for x in range(num_of_datapoints):
        val = f(x)
        if variance != 0:
            y_arr.append(val + random.uniform(-variance*val, variance*val))
        else:
            y_arr.append(val)
    x_arr = [i for i in range(num_of_datapoints)]
    d = []
    for i in range(len(x_arr)):
        d.append((x_arr[i], y_arr[i]))
    return d


def sort(data):
    for i in range(len(data)):
        m = i
        for j in range(i, len(data)):
            if data[j][0] < data[m][0]:
                m = j
        data[i], data[m] = data[m], data[i]

    return data

def convert(data):
    return [d[0] for d in data], [d[1] for d in data]

# TODO fix chunk function: losing one data point at the beginning of each chunk arr
def chunk(data, n):
    arr = []
    ii = 0
    if n > len(data)/2:
        n = int(len(data)/2)
    deltaX = float(data[len(data) - 1][0] - data[0][0])/n
    for i in range(n):
        arr.append([])
    for i in range(len(data)):
        if data[i][0] > (ii+1)*deltaX:
            ii+=1
        arr[ii].append(data[i])
    return arr

def dechunk(data):
    arr = []
    for d in data:
        arr+=d
    return arr

mean = lambda arr: sum(arr)/len(arr)

def averageSlope(data):
    return (sum([r[0]*r[1] for r in data]) - mean(convert(data)[1])*sum(convert(data)[0])) / (sum([r[0]**2 for r in data]) - mean(convert(data)[0])*sum(convert(data)[0]))

def interceptEstimate(data):
    return (mean(convert(data)[1]) * sum([r[0]**2 for r in data]) - mean(convert(data)[0]) * sum([r[0]*r[1] for r in data])) / (sum([r[0]**2 for r in data]) - mean(convert(data)[0]) * sum([r[0] for r in data]))

def generateSlopes(data):
    ms = []
    deltaX = float(dechunk(data)[len(dechunk(data)) - 1][0] - dechunk(data)[0][0]) / len(data)
    for i in range(len(data)):
        c = data[i]
        if len(c) > 0:
            ms.append(averageSlope(c))
    return [((i+0.5)*deltaX, ms[i]) for i in range(len(ms))]

def drawLines(data):
    deltaX = float(dechunk(data)[len(dechunk(data)) - 1][0] - dechunk(data)[0][0]) / len(data)
    for i in range(len(data)):
        c = data[i]
        if len(c) > 0:
            plt.plot([i * deltaX, (i + 1) * deltaX], [averageSlope(c) * (i * deltaX) + interceptEstimate(c),
                                                      averageSlope(c) * ((i + 1) * deltaX) + interceptEstimate(c)])

def drawPoints(data):
    xs, ys = convert(dechunk(data))
    plt.scatter(xs, ys, color='brown', s=20)

def drawFunc(f, xs):
    plt.plot(xs, [f(x) for x in xs], color="#ff00ff")

def drawChunks(data):
    deltaX = float(dechunk(data)[len(dechunk(data)) - 1][0] - dechunk(data)[0][0]) / len(data)
    xs, ys = convert(dechunk(data))
    for i in range(len(data)):
        plt.plot([i*deltaX, i*deltaX], [0.9*min(ys) if min(ys) >= 0 else 1.1*min(ys), 1.1*max(ys) if max(ys) >= 0 else 0.9*max(ys)], color="#000000", linewidth=1)

def getData(f, size, chunks, variance=-1):
    return chunk(sort(create_dataset(f, size, variance)), chunks)

style.use('fivethirtyeight')


data = getData(lambda x: (x+1)**2, 100,5, 0.1)

mdata = generateSlopes(data)
mdash = averageSlope(mdata)
bdash = interceptEstimate(mdata)
C = interceptEstimate(data[0])
f = lambda x : mdash/2.0*x**2+bdash*x+C
f_str = "f(x) = "+str(mdash/2.0)+"x^2 + "+str(bdash)+"x + "+str(C)
print(f_str)

drawFunc(f, convert(dechunk(data))[0])
drawChunks(data)
drawLines(data)
drawPoints(data)

plt.show()

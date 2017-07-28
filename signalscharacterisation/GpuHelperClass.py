from numba import autojit, jit
import math
import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer
from numba import cuda
from numba import float64

@cuda.jit('f8(f8[:])', device=True)
def sum_gpu(x):
    s = 0
    for i in range(len(x)):
        s += x[i]
    return s

@cuda.jit('f8(f8[:], f8[:], f8)', device=True)
def least_square(x, y, order):
    sx = sum_gpu(x)
    sy = sum_gpu(y)
    sxy = 0.0
    sx2 = 0.0
    sy2 = 0.0
    for i in range(len(x)):
        sxy += x[i] * y[i]
    for i in range(len(x)):
        sx2 += x[i] * x[i]
    for i in range(len(x)):
        sy2 += y[i] * y[i]
    n = len(x)
    denom = (n * sx2 - sx * sx)
    b = (sy*sx2 - sx*sxy)/denom
    a = (n*sxy - sx*sy)/denom
    agg_fluc = 0.0

    for i in range(len(x)):
        agg_fluc += ((a*x[i] + b - y[i])**2)

    mean_fluc = math.sqrt(agg_fluc/len(x))

    return float64(mean_fluc)


@cuda.jit('f8(f8[:], f8[:])', device=True)
def device_func_gpu(x, y):
    d = least_square(x, y, 2)
    return d


@cuda.jit('void(f8[:], f8[:,:], f8[:])')
def dfa_kernel(x, y, res):
    n = y.shape[0]
    # x = xrange([1, 2, 3]

    startX = cuda.grid(1)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    # gridY = cuda.gridDim.y * cuda.blockDim.y
    for i in range(startX, n, gridX):
        r = device_func_gpu(x, y[i, :])
        res[i] = r

def gpu_calc_dfa(x, d, order):
    flucs = np.zeros(len(d))
    x = np.array(x, dtype=float)
    res_gpu = cuda.to_device(flucs)
    y_gpu = cuda.to_device(d)
    x_gpu = cuda.to_device(x)

    blockdim = (64, 1)
    griddim = (128, 1)

    dfa_kernel[griddim, blockdim](x_gpu, y_gpu, res_gpu)
    res_gpu.to_host()

    return flucs
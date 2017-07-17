import numpy as np
from timeit import default_timer as timer

fft = np.fft

def autocorrelation(x):
    """
    Compute autocorrelation using FFT
    The idea comes from 
    http://dsp.stackexchange.com/a/1923/4363 (Hilmar)
    """
    x = np.asarray(x)
    N = len(x)
    x = x-x.mean()
    s = fft.fft(x, N*2-1)
    result = np.real(fft.ifft(s * np.conjugate(s), N*2-1))
    result = result[:N]
    result /= result[0]
    return result

def AutoCorrelation(x):
    x = np.asarray(x)
    y = x-x.mean()
    result = np.correlate(y, y, mode='full')
    result = result[len(result)//2:]
    result /= result[0]
    return result 

def autocorrelate(x):
    fftx = fft.fft(x)
    fftx_mean = np.mean(fftx)
    fftx_std = np.std(fftx)

    ffty = np.conjugate(fftx)
    ffty_mean = np.mean(ffty)
    ffty_std = np.std(ffty)

    result = fft.ifft((fftx - fftx_mean) * (ffty - ffty_mean))
    result = fft.fftshift(result)
    return [i / (fftx_std * ffty_std) for i in result.real]


np.set_printoptions(precision=3, suppress=True)

"""
These tests come from
http://www.maplesoft.com/support/help/Maple/view.aspx?path=Statistics/AutoCorrelation
http://www.maplesoft.com/support/help/Maple/view.aspx?path=updates%2fMaple15%2fcomputation
"""
tests = [
    ([1,2,1,2,1,2,1,2], [1,-0.875,0.75,-0.625,0.5,-0.375,0.25,-0.125]),
    ([1,-1,1,-1], [1, -0.75, 0.5, -0.25]),
    ]

for x, answer in tests:
    x = np.array(x)
    answer = np.array(answer)
    # print(autocorrelate(x))
    s = timer()
    for i in range(10):
        autocorrelation(x)
    print(timer() - s)
    s = timer()
    for i in range(10):
        AutoCorrelation(x)
    print(timer() - s)

    assert np.allclose(AutoCorrelation(x), answer)
    print

"""
Test that autocorrelation() agrees with AutoCorrelation()
"""
for i in range(1000):
    x = np.random.random(np.random.randint(2,100))*100
    assert np.allclose(autocorrelation(x), AutoCorrelation(x))
#! /usr/bin/env python
import numpy as np

def spline_kernel(x):
    """
    Spline kernel function. This is definied such that spline_kernel(x=0) = 1, 
    spline_kernel(|x|>1) = 0 and int from -1 to 1 = 3/4.
    """
    if hasattr(x, "__len__"):
        # x is an array
        y = np.zeros(len(x))
        absx = abs(x)
        ind = absx < 0.5
        y[ind] = 1 - 6*absx[ind]**2 + 6*absx[ind]**3
        ind = np.logical_and(absx >= 0.5, absx < 1.)
        y[ind] = 2*(1-absx[ind])**3
    else:
        # x is a number
        absx = abs(x)
        if   absx < 0.5: y = 1 - 6*absx**2 + 6*absx**3
        elif absx < 1:   y = 2*(1-absx)**3
        else:            y=0
    return y
    

def scaled_spline_kernel(x, mean=0, sig=1):
    """
    Spline kernel function which has been rescaled to have a specified mean
    and standard deviaton, and normalized
    """
    y = spline_kernel((x-mean)/(sig*np.sqrt(12)))
    return y / (0.75 * sig * np.sqrt(12))


def spline_kernel_integral(x):
    """
    Returns the integral of the unscaled spline kernel function from -1 to x
    """
    if hasattr(x, "__len__"):
        # x in an array
        integral = np.zeros(len(x))
        absx = abs(x)
        ind = absx < 0.5
        integral[ind] = absx[ind] - 2*absx[ind]**3 + 1.5*absx[ind]**4
        ind = np.logical_and(absx >= 0.5, absx < 1.)
        integral[ind] = 0.375 - 0.5*(1-absx[ind])**4
        ind = absx >= 1.
        integral[ind] = 0.375
        ind = x < 0
        integral[ind] = -integral[ind]
    else:
        # x is a number
        absx = abs(x)
        if   absx < 0.5: integral = absx - 2*absx**3 + 1.5*absx**4
        elif absx < 1:   integral = 0.375 - 0.5*(1-absx)**4
        else:            integral = 0.375
        if x < 0: integral = -integral
    return integral


def cumulative_spline_kernel(x, mean=0, sig=1):
    """
    Returns the integral of the rescaled spline kernel function from -inf to x.
    The spline kernel is rescaled to have the specified mean and standard
    deviation, and is normalized.
    """
    integral = spline_kernel_integral((x-mean)/(sig*np.sqrt(12))) / 0.75
    y = 0.5 * (1. + 2*integral)  
    return y


def _get_length(arr):
    if hasattr(arr, "__len__"): return len(arr)
    else: return np.NaN


def random(size=np.NaN, mean=0, sig=1):
    """
    Draw random numbers from a spline kernel probability distribution of a 
    certain mean and standard deviaton. If size is not specified, a single
    number is returned. If size is specified, an array of random numbers of 
    length size is returned. mean and sig can also be arrays.
    """
    sizes = np.array([size, _get_length(mean), _get_length(sig)])
    if np.isnan(sizes).all():
        size = np.NaN
    elif np.isnan(sizes).any():
        size = int(np.max(sizes[np.invert(np.isnan(sizes))]))
    else:
        size = int(np.max(sizes))

    # check if mean or sig are arrays, check if they are the same length as size
    if (sizes[np.invert(np.isnan(sizes))] < size).any():
        raise ValueError("size mismatch")
    
    # if an array of random numbers is being created, convert mean/sig to arrays
    if not np.isnan(size):
        if np.isnan(sizes[1]): mean = np.ones(size, dtype="f") * mean
        if np.isnan(sizes[2]): sig = np.ones(size, dtype="f") * sig

    if size is np.NaN:
        # generate single random number
        value=0; u=1
        while u > value:
            x = np.random.rand() * 2 - 1
            value = spline_kernel(x)
            u = np.random.rand()
    elif size >= 0:
        # generate array of random numbers
        x = np.random.rand(size) * 2 - 1
        value = spline_kernel(x)
        u = np.random.rand(size)
        index = np.where(u > value)[0] #index of numbers to reject
        while len(index) > 0:
            y = np.random.rand(len(index)) * 2 - 1
            x[index] = y
            value = spline_kernel(y)
            u = np.random.rand(len(index))
            index2 = np.where(u > value)[0]
            index = index[index2]

    return (x * sig * np.sqrt(12)) + mean

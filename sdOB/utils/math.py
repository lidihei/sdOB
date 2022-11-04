import numpy as np

def weighted_average(x, err, w=None):
    '''
    calculate the weigthed arerage, weight, w_i = \frac{1/err_i^2}{sum{1\err_i^2}}
    parameters:
    -------------------------------
    x: [1D array]
    err: [1D array] 
    w: [1D array] the weight, if w is None: w = 1/err**2
    returns:
    ------------------------------
    x_average
    err_average
    '''
    if w is None: w = 1/err**2
    wsum = np.nansum(w)
    wi = w/wsum
    x_average = np.nansum(wi*x)
    err_average = np.sqrt(np.nansum((wi*err)**2))
    return x_average, err_average

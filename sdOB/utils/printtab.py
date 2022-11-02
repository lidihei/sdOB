import numpy as np

def printpm(x, error, precision=3):
    '''
    returns: stri   (x \pm error)
    '''
    num = precision
    stri = '$'f'{x:.{num}f}''\pm'f'{error:.{num}f}''$'
    #print(stri)
    return stri

def printmlu(x, precision=3, percentile=[16, 50, 84]):
    '''
    returns: stri   (x^{+err}_{-err})
    '''
    num = precision
    x = x[~np.isnan(x)]
    xp = np.percentile(x, percentile)
    l,u = np.diff(xp)
    stri = '$'f'{xp[1]:.{num}f}''_{-'f'{l:.{num}f}''}''^{+'f'{u:.{num}f}''}$'
    #print(stri)
    return stri

def printmstd3(x, precision=3):
    '''
    returns: stri   (x \pm error)
    '''
    num = precision
    xm = np.mean(x)
    error = np.std(x)
    stri = '$'f'{xm:.{num}f}''\pm'f'{error:.{num}f}''$'
    #print(stri)
    return stri

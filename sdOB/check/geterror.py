def medianstd(x):
    xp = np.percentile(x, [16, 50, 84])
    sigma = np.sqrt(np.sum(np.diff(xp)**2)/2)
    return xp[1], sigma

def printmlu(x):
    x = x[~np.isnan(x)]
    xp = np.percentile(x, [16, 50, 84])
    l,u = np.diff(xp)
    stri = '$'f'{xp[1]}''_{-'f'{l}''}''^{+'f'{u}''}$'
    print(stri)
    return stri

def printpm(x, error):
    stri = '$'f'{x}''\pm'f'{error}''$'
    print(stri)
    return stri



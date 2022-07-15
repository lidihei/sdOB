# calculate equibalent width

import numpy as np
from scipy.optimize import curve_fit

class calEW():
    
    def gaussian(self, x, A, mu, sigma):
        y = A/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2*sigma**2))
        return y
    
    def linearfunc(self, x, k,b):
        y = k*x +b
        return y
    
    def sersic(self, x, A, mu, sigma, c):
        return A*np.exp(-(np.abs(x-mu)/np.abs(sigma))**c)
    
    def sersic_base(self, x, A, mu, sigma, c, b):
        return A*np.exp(-(np.abs(x-mu)/np.abs(sigma))**c)+b
    
    def sersic_linear(self, x,  A, mu, sigma, c, k, b):
        y = self.gaussian(x, A, mu, sigma) + k*x +b
        return y
    
    def gauss_linear(self, x, A, mu, sigma, k, b):
        y = self.gaussian(x, A, mu, sigma) + k*x +b
        return y
    
    def gauss2_linear(self, x, A1, mu1, sigma1, A2, mu2, sigma2,  k, b):
        if mu2 < mu1: return x*np.inf
        y = self.gaussian(x, A1, mu1, sigma1) + k*x +b
        y +=  self.gaussian(x, A2, mu2, sigma2)
        return y

    def gauss3_linear(self, x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, k, b):
        if mu2 < mu1: return x*np.inf
        if mu3 < mu2: return x*np.inf
        y = self.gaussian(x, A1, mu1, sigma1) + k*x +b
        y +=  self.gaussian(x, A2, mu2, sigma2)
        y +=  self.gaussian(x, A3, mu3, sigma3)
        return y

    def fit_func(self, xobs, yobs, p0, func=None, percentile=99, inters=3,sigma=None,  **kwargs):
        popt,pcov = curve_fit(func,xobs,yobs, p0=p0, sigma=sigma, **kwargs)
        for _ in np.arange(3):
           y = func(xobs, *popt)
           dy = np.abs(yobs-y)
           ind = dy<np.percentile(dy,percentile)
           popt,pcov = curve_fit(func,xobs[ind],yobs[ind], p0=p0, **kwargs)
        self.popt = popt
        self.pcov = pcov
        self.func = func
        self.xobs = xobs
        self.yobs = yobs
        self.yerrobs = sigma
        return popt,pcov
    
    def get_multivariate_normal(self, popt, pcov, size=100):
        return np.random.multivariate_normal(popt, pcov, size).T
    
    def _EW_gauss_linear(self, A, mu, sigma, k, b, x=None):
        if x is not None:
           y = -self.gaussian(x, A, mu, sigma)/(k*x +b)
           ew = np.trapz(y, x=x)
        else:
           ew = -A/(k*mu +b)
           y = 0
        return ew, y
    
    def EW_gauss_linear(self, As, mus, sigmas, ks, bs, x=None):
        '''
        As, mus, sigmas, ks, bs have the same length
        parameters:
        ---------------
        As: [1D array]
        mus: [1D array]
        sigmas: [1D array]
        ks: [1D array]
        bs: [1D array]
        x: [1D array]
        returns: 
        ew
        ew_err
        '''
        ews = np.zeros_like(As)
        for _i, (A, mu, sigma, k, b) in enumerate(zip(As, mus, sigmas, ks, bs)):
            ews[_i], _ = self. _EW_gauss_linear(A, mu, sigma, k, b, x=x)
        ew = np.mean(ews)
        ew_err = np.std(ews)
        return ew, ew_err
    
    def dlambda2rv(self, lam, lamerr, lam0):
        '''
        parameters:
        -------------
        lam: [float] the wavelength of emission or absorption lines (in angstrom)
        lamerr: [float] the error of the line
        lam0: the rest reference wavelength of the line
        returns:
        ----------
        rv: [float] velocity (km/s)
        rverr: [float] velocity error (km/s)
        '''
        c = 299792.458
        rv = (lam - lam0)/lam0 * c
        rverr = lamerr*c/lam0
        return rv, rverr
    
    def getP0_gauss_linear(self, xobs, yobs, mu=None):
        if mu is None: mu = xobs[np.argmin(yobs)]
        y = (xobs-mu)**2
        sigma = np.sqrt(np.trapz(y, x=xobs)/np.trapz(yobs, x=xobs))
        k = (yobs[0] - yobs[-1]) / (xobs[0] - xobs[-1])
        b = -k*xobs[0]+yobs[0]
        A = np.argmin(yobs) - (k*mu+b)
        p0 = [A, mu, sigma, k, b]
        self.p0 = p0
        return p0
    
    def plotresault(self, xdens=None):
        import matplotlib.pyplot as plt
        xobs = self.xobs
        yobs = self.yobs
        yerrobs = self.yerrobs
        popt = self.popt
        if xdens is None: xdens = np.linspace(np.min(xobs), np.max(xobs), 10000)
        ydens = self.func(xdens, *popt)
        y = self.func(xobs, *popt)
        dy = yobs -y
        fig, axs = plt.subplots(2,1,figsize=[7,6], gridspec_kw={'height_ratios': [2,1]}, sharex=True)
        plt.subplots_adjust(left=0.15,top=0.97, right=0.97, bottom=0.15, hspace=0)
        plt.sca(axs[0])
        if yerrobs is not None:
           plt.errorbar(xobs, yobs, yerr=yobserr, color='k', fmt='o')
        else:
           plt.scatter(xobs, yobs, color='k')
        plt.plot(xdens, ydens, color='r')
        plt.sca(axs[1])
        if yerrobs is not None:
           plt.errorbar(xobs, dy, yerr=yobserr, color='k', fmt='o')
        else:
           plt.scatter(xobs, dy, color='k')
        self.fig = fig
        self.axs = axs

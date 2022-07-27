# calculate equibalent width

import numpy as np
from scipy.optimize import curve_fit
from laspec import normalization
from astropy.io import fits
from  . import spec_tools as spect
from . import time_tools as timet
from astropy import units, constants
from astropy.table import Table

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

    def fit_func(self, xobs, yobs, p0, func=None, percentile=99, inters=3,yerrobs=None,  **kwargs):
        popt,pcov = curve_fit(func,xobs,yobs, p0=p0, **kwargs)
        if inters is not None:
           for _ in np.arange(inters):
              y = func(xobs, *popt)
              dy = np.abs(yobs-y)
              ind = dy<np.percentile(dy,percentile)
              popt,pcov = curve_fit(func,xobs[ind],yobs[ind], p0=p0, **kwargs)
        self.popt = popt
        self.pcov = pcov
        self.func = func
        self.xobs = xobs
        self.yobs = yobs
        self.yerrobs = yerrobs
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



class ew_LBY(calEW):
    
    ''' calculate EW of spectra observed by LAMOST, BFOSC and YFOSC
    '''
    def read_fosc_spec(self, fname):
        '''
        returns:
        -------------
        wave, flux, fluxerr, bjd, rv, rverr
        '''
        hdu = fits.open(fname)
        data = hdu[1].data
        self.hdu = hdu
        wave = 10**data['loglambda']
        wave = 10**spect.rvcorr_spec(wave, wave, wave, hdu[0].header['BARYCORR'], returnwvl=True)
        flux = data['flux']
        fluxerr = data['error']
        bjd = hdu[0].header['bjd']
        rv = hdu[0].header['baryrv']
        rverr = hdu[0].header['rv_err']
        self.fname = fname
        self.bjd = bjd
        self.rv = rv
        self.rverr = rverr
        self.wave = wave
        self.flux = flux
        self.fluxerr = fluxerr
        return wave, flux, fluxerr
    
    def read_lamostlrs_specdr7(self, fname):
        '''
        returns:
        -------------
        wave: [in angstrom]  wavelength in air
        flux
        fluxerr
        '''
        hdu = fits.open(fname)
        wave = hdu[0].data[2]
        flux = hdu[0].data[0]
        fluxerr = np.sqrt(1/ hdu[0].data[1])
        hdu.close()
        header = hdu[0].header
        self.hdu = hdu
        _ra, _dec, date_start, date_end = [header[_] for _ in ('ra', 'dec', 'DATE-BEG', 'DATE-END')]
        bjd, _ =  eval_ltt_lamost(_ra, _dec, date_start, date_end, barycorr=False)
        wave = spect.vacuum2air(wave)
        c = constants.c.to('km/s').value
        self.bjd = bjd
        self.rv = header['Z'] *c
        self.rverr = header["Z_err"]*c
        self.fname = fname
        self.wave = wave
        self.flux = flux
        self.fluxerr = fluxerr
        return wave, flux, fluxerr

    def read_lamostlrs_specdr9(self, fname):
        '''
        returns:
        wave: [in angstrom]  wavelength in air
        flux
        fluxerr
        '''
        hdu = fits.open(fname)
        wave = hdu[1].data['wavelength'][0]
        flux = hdu[1].data['flux'][0]
        fluxerr = np.sqrt(1/hdu[1].data['IVAR'][0])
        wave = spect.vacuum2air(wave)
        flux_norm, flux_cont = normalization.normalize_spectrum_spline(wave, flux, niter=3)
        header = hdu[0].header
        _ra, _dec, date_start, date_end = [header[_] for _ in ('ra', 'dec', 'DATE-BEG', 'DATE-END')]
        bjd, _ =  timet.eval_ltt_lamost(_ra, _dec, date_start, date_end, barycorr=False)
        c = constants.c.to('km/s').value
        self.bjd = bjd
        self.rv = header['Z'] *c
        self.rverr = header["Z_err"]*c
        self.fname = fname
        self.wave = wave
        self.flux = flux
        self.fluxerr = fluxerr
        self.hdu = hdu
        return wave, flux, fluxerr
   
    def normalizeflux(self, wave=None, flux=None, p=1e-06, q=0.5, lu=(-1, 3), binwidth=30, niter=5):
        '''
        parameters:
        details see: from from laspec import normalization; normalization.normalize_spectrum_spline?
        returns
        ------------
        flux_norm: noralized flux
        flux_cont: continue flux 
        '''
        if wave is None: wave = self.wave
        if flux is None: flux = self.flux
        flux_norm, flux_cont = normalization.normalize_spectrum_spline(wave, flux, niter=3)
        self.flux_norm = flux_norm
        self.flux_cont = flux_cont
        return flux_norm, flux_cont

    def getrvtab(self, fname=None, bjd=None, rv =None, rverr=None):
        '''
        returns:
        tab: astropy.table
        '''
        if fname is None: fname = self.fname
        if bjd is None: bjd = self.bjd
        if rv is None: rv = self. rv
        if rverr is None: rverr = self.rverr
        data = [[fname], [bjd], [rv], [rverr]]
        names = ['fname', 'bjd', 'rv', 'rverr']
        tab= Table(data=data, names=names)
        return tab
    
    def get_EW_NaD(self, p0=[-2, 5874., 2, 2, 5892, 2, 2, 5896, 1,  0,  0], wave=None, flux=None, fluxerr = None, show=True):
        if wave is None: wave = self.wave
        if flux is None: flux = self.flux
        if fluxerr is None: fluxerr = self.fluxerr
        _ind =(wave> 5810) & (wave < 6000)
        xobs, yobs, yerrobs = [_[_ind] for _ in [wave, flux, fluxerr]]

        popt,pcov = self.fit_func(xobs, yobs, p0, func=self.gauss3_linear, percentile=99, inters=1)
        x = None # np.linspace(3930, 3944, 1000)
        popts = self.get_multivariate_normal(popt,pcov, size=1000)
        ews = np.zeros((3,2))
        
        ews[0] =self.EW_gauss_linear(*popts[[0,1,2, 9,10]], x=x)
        ews[1] =self.EW_gauss_linear(*popts[[3,4,5, 9,10]], x=x)
        ews[2] =self.EW_gauss_linear(*popts[[6,7,8, 9,10]], x=x)
        rvs = np.zeros((3,2))
        rvs[0] = self.dlambda2rv(popt[1], np.sqrt(pcov[1,1]), 5875.621)
        rvs[1] = self.dlambda2rv(popt[4], np.sqrt(pcov[4,4]), 5889.95095)
        rvs[2] = self.dlambda2rv(popt[7], np.sqrt(pcov[7,7]), 5895.92424)
        if show:
           fig, ax = plt.subplots(1,1,figsize=(7,4))
           plt.plot(wave, flux, color='b', lw=0.7)
           plt.plot(xobs, yobs, '.')
           plt.xlim(5700, 6000)
           plt.axvline(x=5780)
           plt.axvline(x=5875.621)
           plt.axvline(x=5889.95095) # Na I
           plt.axvline(x=5895.92424) # Na I
           plt.show()
           self.plotresault(xdens=None)
           plt.sca(self.axs[0])
           plt.plot(wave, flux, color='b', lw=0.7)
           #plt.title(r'$EW = 'f'{_ew:.4f}\pm{ewerr:.4f}''\AA$\n'r' $rv_{CaII K}$'f' = {rv_cak:.2f}'r'$\pm$'f'{rverr_cak:.2f} km/s')
           plt.xlim(5810, 6000)
        
        linename = ['HeI5876', 'NaD1', 'NaD2']
        line_lam = [5875.621, 5889.95095, 5895.9242]
        _tab = Table(data=[linename, line_lam, ews.T[0], ews.T[1], rvs.T[0], rvs.T[1]], 
                     names=('line', 'lambda_air', 'ew', 'ew_err', 'rv', 'rv_err'),
                    dtype=('<U10', 'f8', '<f4', '<f4', '<f4', '<f4'))
        
        data = np.hstack((ews, rvs)).ravel()
        names = np.ravel([[f'ew_{_}', f'ewerr_{_}', f'rv_{_}', f'rverr_{_}'] for _ in linename ])
        tab= Table(data=data, names=names, dtype=['<f4']*len(data))
        self._tab = _tab
        return tab
    
    def get_EW_CaK(self, p0=[2, 3934, 2,  0,  0], x=None, wave=None, flux=None, fluxerr = None, show=True):
        if wave is None: wave = self.wave
        if flux is None: flux = self.flux
        if fluxerr is None: fluxerr = self.fluxerr
        _ind =( (wave> 3910) & (wave < 3921)) | ((wave> 3930) & (wave < 3945))
        xobs, yobs, yerrobs = [_[_ind] for _ in [wave, flux, fluxerr]]
        popt,pcov = self.fit_func(xobs, yobs, p0, func=self.gauss_linear, percentile=99, inters=1)
        x = x # np.linspace(3930, 3944, 1000)
        popts = self.get_multivariate_normal(popt,pcov, size=1000)
        ews = np.zeros((1,2))
        
        ews[0] =self.EW_gauss_linear(*popts[[0,1,2, 9,10]], x=x)
        rvs = np.zeros((1,2))
        rvs[0] = self.dlambda2rv(popt[1], np.sqrt(pcov[1,1]), 3933.66)
        
        linename = ['CaII K']
        line_lam = [3933.66]
        _tab = Table(data=[linename, line_lam, ews.T[0], ews.T[1], rvs.T[0], rvs.T[1]], 
             names=('line', 'lambda_air', 'ew', 'ew_err', 'rv', 'rv_err'),
            dtype=('<U10', 'f8', '<f4', '<f4', '<f4', '<f4'))
        names = np.ravel([[f'ew_{_}', f'ewerr_{_}', f'rv_{_}', f'rverr_{_}'] for _ in linename ])
        tab= Table(data=data, names=names, dtype=['<f4']*len(data))
        self._tab = _tab
        if show:
           fig, ax = plt.subplots(1,1,figsize=(7,4))
           plt.plot(wave, flux, color='b')
           plt.plot(xobs, yobs, '.')
           plt.axvline(x=3934)
           plt.xlim(3900, 4000)
           #plt.ylim(17, 24)
           plt.show() 
           self.plotresault(xdens=None)
           plt.sca(self.axs[0])
           plt.plot(wave, flux, color='b', lw=0.7)
           #plt.title(r'$EW = 'f'{_ew:.4f}\pm{ewerr:.4f}''\AA$\n'r' $rv_{CaII K}$'f' = {rv_cak:.2f}'r'$\pm$'f'{rverr_cak:.2f} km/s')
           plt.xlim(3900, 4000)
        return tab




        

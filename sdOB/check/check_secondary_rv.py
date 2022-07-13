import scipy
import radvel, ellc
import lightkurve as lk
from astropy import constants, units
import joblib
import matplotlib.pyplot as plt
from ..utils.spec_tools import get_loglam, rvcorr_spec
from speedysedfit.broadsed import vgconv, rotconv, interp_spl
from laspec import normalization
import numpy as np

'''
check mcmc resault of secondary radial velocity
'''


class check_rv():
        
    def pymcsresualt(self, trace, rv1 = None, vsini1=None, q=None, vgamma=None, vsini2=None, R2tR1=None, rv2=None):
        self.trace = trace
        self.rv1s = trace.posterior.rv1.values.ravel() if (rv1 is None) else rv1
        self.vsini1s = trace.posterior.vsini1.values.ravel() if (vsini1 is None) else vsini1
        self.qs = trace.posterior.q.values.ravel() if (q is None) else q
        self.vsini2s = trace.posterior.vsini2.values.ravel() if (vsini2 is None) else vsini2
        self.R2tR1s = trace.posterior.R2tR1.values.ravel() if (R2tR1 is None) else R2tR1
        self.vgammas = trace.posterior.vgamma.values.ravel() if (vgamma is None) else vgamma 
        self.rv2s = self.vgammas - self.qs*(self.rv1s-self.vgammas) if (rv2 is None) else rv2
        self.bestind = np.argmax(self.trace.log_likelihood.likelihood.values.ravel())
        
    def print_best_values(self, bestind=True):
        #---------------------------best point------------------------------------------------------
        argmax = self.bestind
        _rv1= self.rv1s[argmax] if bestind else np.median(self.rv1s)
        _rv2= self.rv2s[argmax] if bestind else np.median(self.rv2s)
        _vgamma= self.vgammas[argmax] if bestind else np.median(self.vgammas)
        _q= self.qs[argmax] if bestind else np.median(self.qs)
        _vsini1= self.vsini1s[argmax] if bestind else np.median(self.vsini1s)
        _vsini2= self.vsini2s[argmax] if bestind else np.median(self.vsini2s)
        _R2tR1= self.R2tR1s[argmax] if bestind else np.median(self.R2tR1s)
        print('----------------------best values----------------------------------')
        print(f'rv1, rv2, vgamma, 1 = {_rv1}, {_rv2}, {_vgamma}, {_q} ')
        print(f'vsini1, vsini2  = {_vsini1}, {_vsini2}')
        print('-----------------------------------------------------------------')
    
    def get_flux_syn(self, rv1, vsini1, rv2, vsini2, R2tR1, resolution=8000,\
                 logT1=None, logg1=None, logT2=None, logg2=None, Z1=0.38, Z2 = 0.38,\
                 flux1=None, flux2=None, slmodel='None', epsilon1=0.2, epsilon2=0.2, **kwargs):
        '''
        synthesize spectrum with parameters: rv1, vsini1, rv2, vsini2, R2tR1, logT1, logg1, logT2, logg2
        paremeters:
        ------------------
        wavelength: [array] synthetic wave of slam model
        flux1: the synthectic flux of star1
        flux2: the synthectic flux of star2
        slmodel: slam trained model
        returns:
        ------------------
        wbin [array] broaded wave by using instruments resolution
        nflux [array] normalized broaded flux 
        '''
        if flux1 is None:
           flux1 = np.exp(slmodel.predict_spectra(np.array([logT1, logg1, Z1]))[0])
        if flux2 is None:
           flux2 = np.exp(slmodel.predict_spectra(np.array([logT2, logg2, Z2]))[0])
        wave_slm = slmodel.wave
        wave = 10**get_loglam(30000, wave_slm[1], wave_slm[-2])
        flux1, _ = rvcorr_spec(wave_slm, flux1, flux1,rv1, wave_new=wave, left=np.nan, right=np.nan, interp1d=None)
        flux2, _ = rvcorr_spec(wave_slm, flux2, flux2,rv2, wave_new=wave, left=np.nan, right=np.nan, interp1d=None)
        R1 = 1
        R2 = R2tR1
        if vsini1 != 0:
            wave1_vsini, flux1_vsini = rotconv(wave, flux1, epsilon1, vsini1)
        else:
            wave1_vsini, flux1_vsini = wave, flux1
        if vsini2 != 0:
            wave2_vsini, flux2_vsini = rotconv(wave, flux2, epsilon2, vsini2)
        else:
            wave2_vsini, flux2_vsini = wave, flux2
        flux2, _ = rvcorr_spec(wave2_vsini, flux2_vsini, np.zeros(len(wave2_vsini)),\
                                    0, wave_new=wave, left=1, right=1, interp1d=None)
        flux1, _ = rvcorr_spec(wave1_vsini, flux1_vsini, np.zeros(len(wave1_vsini)),\
                                        0, wave_new=wave, left=1, right=1, interp1d=None)
        flux = R1**2*flux1 + R2**2*flux2
        def getind(flux):
            ind = ~np.isnan(flux)
            return ind

        _ind = ~np.isnan(flux)
        wave = wave[_ind]
        flux = flux[_ind]
        flux1 = flux1[_ind]
        flux2 = flux2[_ind]
        RR =  resolution # spectrum resolution
        clight = 299792.458 # km/s
        vfhmw = clight/RR
        wbin, fbin = vgconv(wave, flux,vfhmw)
        _, fbin1 = vgconv(wave, flux1,vfhmw)
        _, fbin2 = vgconv(wave, flux2,vfhmw)
        wave2 = 10**get_loglam(RR, wave_slm[1], wave_slm[-2])
        fbin, _ = rvcorr_spec(wave, fbin, fbin, 0, wave_new= wave)
        fbin1, _ = rvcorr_spec(wave, fbin1, fbin1, 0, wave_new= wave)
        fbin2, _ = rvcorr_spec(wave, fbin2, fbin2, 0, wave_new= wave)
        nflux = normalization.normalize_spectrum_spline(wbin, fbin, niter=5)
        nflux1 = normalization.normalize_spectrum_spline(wbin, fbin1, niter=5)
        nflux2 = normalization.normalize_spectrum_spline(wbin, fbin2, niter=5)
        self.wave = wbin
        self._flux = (fbin, nflux[0], nflux[1])
        self._flux1 = (fbin1, nflux1[0], nflux1[1])
        self._flux2 = (fbin2, nflux2[0], nflux2[1])
        self.normflux = nflux[0]
        self.normflux1 =(R1**2 *fbin1 + R2**2*nflux2[1])/nflux[1]
        self.normflux2 =(R1**2 *nflux1[1] + R2**2*fbin2)/nflux[1]
        return wbin, nflux[0]

    def plotconer(self, samples=None, labels=None):
        import corner
        if samples is None:
           samples = np.vstack((self.rv1s, self.vsini1s, self.rv2s, self.vsini2s, self.R2tR1s)).T
        if labels is None: labels = [r'$rv_{1}$', r'$vsini_{1}$', '$rv_{2}$',  r'$vsini_{2}', '$R_{2}/R_{1}$']
        fig = corner.corner(
                samples,
                labels=labels,
                hist_kwargs={"density": True},
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
            )

    def check_spec(self, wave, flux, rv1=None, vsini1=None, rv2=None, vsini2=None, R2tR1=None, show=True, bestind=False, figax=None,  **kwargs):
        '''
        slmodel: [object] a slam trained model
        returns:
        slamspec:[array] 
        -----------
        '''
        if rv1 is None: rv1 = self.rv1s[self.bestind] if bestind else np.median(self.rv1s)
        if vsini1 is None: vsini1 = self.vsini1s[self.bestind] if bestind else np.median(self.vsini1s)
        if rv2 is None: rv2 = self.rv2s[self.bestind] if bestind else np.median(self.rv2s)
        if vsini2 is None: vsini2 = self.vsini2s[self.bestind] if bestind else np.median(self.vsini2s)
        if R2tR1 is None: R2tR1 = self.R2tR1s[self.bestind] if bestind else np.median(self.R2tR1s)
        waveslam, slamspec = self.get_flux_syn(rv1, vsini1, rv2, vsini2, R2tR1, **kwargs)
        self.slamspec = slamspec
        self.R2tR1 = R2tR1
        if show:
          
           #fig, axs = plt.subplots(2,1, figsize=(15, 4), sharex=True, gridspec_kw={'height_ratios': [2,1]})
           #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0)
           fig, ax = plt.subplots(1,1, figsize=(15, 4)) if figax is None else figax
           e1, = plt.plot(wave, flux, 'b')
           e2, = plt.plot(waveslam, slamspec, 'y')
           leg = plt.legend([e1, e2], ['observed', 'slam'])
           plt.ylabel('normalized flux')
           plt.xlabel('wavelength [$\AA$]')
           for line, text in zip(leg.get_lines(), leg.get_texts()):
               text.set_color(line.get_color())
           self.slamfig = fig
           self.slamax = ax
        return slamspec

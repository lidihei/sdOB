import numpy as np
from pyrafspec.spec_tools import rvcorr_spec
from pyrafspec.broadsed import broadspc, vgconv,rotconv
from laspec import normalization


class SB2_spectra():

    def __init__(self, wave1, flux1, wave2, flux2):
        '''
        generate a SB2 of binary
        paramters:
        ------------------------
        wave1: [1D array] the sythetic wave of star1
        flux1: [1D array] the sythetic flux of star1
        wave2: [1D array] the sythetic wave of star2
        flux2: [1D array] the sythetic flux of star2
        '''
        self.wave1 = wave1
        self.flux1 = flux1
        self.wave2 = wave2
        self.flux2 = flux2

    def get_flux_syn(self, rv1, vsini1, R1, rv2, vsini2, R2, waveobs=None, resolution=8000,\
                     epsilon1=0.2, epsilon2=0.2, resolution0 = 300000, **kwargs):
        '''
        synthesize spectrum with parameters: rv1, vsini1, rv2, vsini2, R2tR1, logT1, logg1, logT2, logg2
        paremeters:
        ------------------
        rv1, vsini1, R1 [int] are radial velolcity, vsini and radius of star1
        rv2, vsini2, R2 [int] are radial velolcity, vsini and radius of star1
        waveobs [1D array]
        returns:
        ------------------
        wbin [array] broaded wave by using instruments resolution
        nflux [array] normalized broaded flux 
        '''
        if waveobs is None:
           w1 = np.max([np.min(self.wave1), np.min(self.wave2)])
           w2 = np.min([np.max(self.wave1), np.max(self.wave2)])
           _ind1 = (self.wave1 > w1) & (self.wave1 < w2)
           _ind2 = (self.wave2 > w1) & (self.wave2 < w2)
           n1 = np.sum(_ind1)
           n2 = np.sum(_ind2)
           if n1 < n2:
              waveobs = self.wave1[_ind1]
           else: waveobs = self.wave2[_ind2]
        flux1_rv, _ = rvcorr_spec(self.wave1, self.flux1, self.flux1, rv1, wave_new=waveobs, left=np.nan, right=np.nan, interp1d=None)
        flux2_rv, _ = rvcorr_spec(self.wave2, self.flux2, self.flux2, rv2, wave_new=waveobs, left=np.nan, right=np.nan, interp1d=None)
        if vsini1 != 0:
            wave1_vsini, flux1_vsini = rotconv(waveobs, flux1_rv, epsilon1, vsini1)
        else:
            wave1_vsini, flux1_vsini = waveobs, flux1_rv
        if vsini2 != 0:
            wave2_vsini, flux2_vsini = rotconv(waveobs, flux2_rv, epsilon2, vsini2)
        else:
            wave2_vsini, flux2_vsini = waveobs, flux2_rv
        flux2, _ = rvcorr_spec(wave2_vsini, flux2_vsini, np.zeros(len(wave2_vsini)),\
                                    0, wave_new=waveobs, left=1, right=1, interp1d=None)
        flux1, _ = rvcorr_spec(wave1_vsini, flux1_vsini, np.zeros(len(wave1_vsini)),\
                                        0, wave_new=waveobs, left=1, right=1, interp1d=None)
        flux = R1**2*flux1 + R2**2*flux2
        def getind(flux):
            ind = ~np.isnan(flux)
            return ind

        _ind = ~np.isnan(flux)
        wave = waveobs[_ind]
        flux = flux[_ind]
        flux1 = flux1[_ind]
        flux2 = flux2[_ind]
        RR =  resolution # spectrum resolution
        clight = 299792.458 # km/s
        fwhm = clight/RR
        fwhm0 = clight/resolution0
        vfwhm = np.sqrt(fwhm**2 - fwhm0**2)
        wbin, fbin = vgconv(wave, flux,vfwhm)
        wbin1, fbin1 = vgconv(wave, flux1,vfwhm)
        wbin2, fbin2 = vgconv(wave, flux2,vfwhm)
        fbin, _ = rvcorr_spec(wbin, fbin, fbin, 0, wave_new= wave)
        fbin1, _ = rvcorr_spec(wbin1, fbin1, fbin1, 0, wave_new= wave)
        fbin2, _ = rvcorr_spec(wbin2, fbin2, fbin2, 0, wave_new= wave)
        nflux = normalization.normalize_spectrum_spline(wave, fbin, niter=5)                                                                                              
        nflux1 = normalization.normalize_spectrum_spline(wave, fbin1, niter=5)
        nflux2 = normalization.normalize_spectrum_spline(wave, fbin2, niter=5)
        self.wave = wave
        self.flux = flux
        self.flux1_broad = fbin1
        self.flux2_broad = fbin2
        self._flux = (wave, nflux[0], nflux[1])
        self._flux1 = (wave, nflux1[0], nflux1[1])
        self._flux2 = (wave, nflux2[0], nflux2[1])
        self.normflux = nflux[0]
        self.normflux1 =(R1**2 *fbin1 + R2**2*nflux2[1])/nflux[1]
        self.normflux2 =(R1**2 *nflux1[1] + R2**2*fbin2)/nflux[1]
        return wave, nflux[0]


if __name__ == '__main__':
    from pyrafspec.splicing_spectrum import get_loglam
    import matplotlib.pyplot as plt
    import joblib
    '''
    wave = 10**get_loglam(300000, 4000, 8000)
    flux = 1 - 0.4*np.exp(-(wave - 6000)**2/4**2)
    sb2spec = SB2_spectra(wave, flux, wave, flux)
    sb2spec.get_flux_syn(500, 0, 1, -200, 0, 0.8, waveobs=wave, resolution=20000)
    plt.plot(sb2spec.wave, sb2spec.normflux, color='k')
    plt.plot(sb2spec.wave, sb2spec.normflux1, label='star1', color='r')
    plt.plot(sb2spec.wave, sb2spec.normflux2, label='star2', color='b')
    plt.legend()
    plt.show()
    '''
    regli_tlusty = joblib.load('/share/data/lijiao/Documents/sdOB/example/Feige64//data/regli_spec/TLUSTY_grid_Feige64.z')['Regli']
    flux_star1 = regli_tlusty.interpn([np.log10(15618), 3.87, 0.72])
    flux_star2 = regli_tlusty.interpn([np.log10(15000), 4, 0.72])
    sb2spec = SB2_spectra(regli_tlusty.wave, flux_star1, regli_tlusty.wave, flux_star2)
    waveobs = 10**get_loglam(3000, 6000, 7100)
    sb2spec.get_flux_syn(60, 50, 0.87, -10, 0, 1, waveobs =waveobs,  resolution=20000)

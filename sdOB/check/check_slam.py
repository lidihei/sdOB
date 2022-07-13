from astropy.table import Table
import scipy
import radvel, ellc
import lightkurve as lk
from astropy import constants, units
from radvel.orbit import timeperi_to_timetrans, timetrans_to_timeperi
from ..utils.lc_tools import *
from ..utils.lyd import *
import joblib
import matplotlib.pyplot as plt


'''
check mcmc resault of slam
'''


class check_slam():
        
    def pymcsresualt(self, trace, logTe=None, logg=None, Z=None, vsini=None):
        self.trace = trace
        self.logTes = trace.posterior.logTe.values.ravel() if (logTe is None) else logTe
        self.loggs = trace.posterior.logg.values.ravel() if (logg is None) else logg
        self.Zs = trace.posterior.Z.values.ravel() if (Z is None) else Z
        self.vsinis = trace.posterior.vsini.values.ravel() if (vsini is None) else vsini
        self.bestind = np.argmax(self.trace.log_likelihood.likelihood.values.ravel())
        
    def print_best_values(self):
        #---------------------------best point------------------------------------------------------
        argmax = self.bestind
        _logTe= self.logTes[argmax]
        _logg= self.loggs[argmax]
        _Z= self.Zs[argmax]
        _vsini= self.vsinis[argmax]
        print('----------------------best values----------------------------------')
        print(f'logTe, logg, Z, vsini = {_logTe}, {_logg}, {_Z}, {_vsini} ')
        print(f'Teff  = {10**_logTe}')
        print('-----------------------------------------------------------------')
            
    def check_spec(self, wave, flux, ind, slmodel, logTe=None, logg=None, Z=None,vsini=None, show=True, bestind=False):
        '''
        slmodel: [object] a slam trained model
        returns:
        slamspec:[array] 
        -----------
        '''
        if logTe is None: logTe = self.logTes[self.bestind] if bestind else np.median(self.logTes)
        if logg is None: logg = self.loggs[self.bestind] if bestind else np.median(self.loggs)
        if Z is None: Z = self.Zs[self.bestind] if bestind else np.median(self.Zs)
        if vsini is None: vsini = self.vsinis[self.bestind] if bestind else np.median(self.vsinis)
        Xpred = np.array([logTe, logg, Z, vsini])
        slamspec = slmodel.predict_spectra(Xpred)[0]
        self.slamspec = slamspec
        dy =flux-slamspec
        _ind =  ~np.isnan(dy)
        if show:
           fig, axs = plt.subplots(2,1, figsize=(15, 7), sharex=True, gridspec_kw={'height_ratios': [2,1]})
           plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0)
           e1, = axs[0].plot(wave, flux)
           e2, = axs[0].plot(wave, slamspec)
           axs[1].plot(wave, dy, c='grey', alpha=0.5)
           axs[1].scatter(wave[ind], dy[ind], color='b', s=1, alpha=0.5)
           plt.sca(axs[0])
           leg = plt.legend([e1, e2], ['observed', 'slam'])
           plt.ylabel('normalized flux')
           plt.sca(axs[1])
           plt.ylabel('$f_{obs} - f_{slam}$')
           plt.xlabel('wavelength [$\AA$]')
           for line, text in zip(leg.get_lines(), leg.get_texts()):
               text.set_color(line.get_color())
        self.slamfig = fig
        self.slamax = axs
        return slamspec

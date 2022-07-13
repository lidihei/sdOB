from laspec.extern.interpolate import SmoothSpline
from laspec import convolution
from PyAstronomy import pyasl
import numpy as np


def reinterp_wave(ipar, Tpars, Tlusty_sps, wave0, wave, linearinterp=False):
    '''interpolate spectrum in a new wavelength
    parameters:
    ------------------
    ipar [int]
    Tpars [ 2D array] parameters array of TLUSTY grid e.g. np.array([[teff, logg, z, vt]])
    Tlusty_sps [2D array] spectra array of TLUSTY grid
    wave0 [1D array] the original wavelength of TLUSTY spectra
                     e.g. BTLUSTY optical: wave = np.arange(3200.01, 9998.7, 0.01); 
                          OTLUSTY: wave = np.arange(3000, 7500, 0.01)
    wave [1D array] the new wavelength
    returns:
    -------------
    pars [1D array] which has the same parameters with Tpars
    spec [1D array] has same length as wave
    '''
    #print(len(Tlusty_sps[ipar]), len(wave0))
    if linearinterp:
        spec = np.interp(wave, wave0, Tlusty_sps[ipar])
    else:
        func = SmoothSpline(wave0, Tlusty_sps[ipar], p=1)
        spec = func(wave)
    return Tpars[ipar], spec


def smooth_by_bin(time, flux, bins):
    '''smooth light curve by bins
    parameters:
    -------------------
    time: [array] could be phase
    flux: [flux]
    bins: [array]
    returns:
    -------------------
    time_bins: [array] time_bins = (bins[1:] + bins[:-1])/2
    flux_smooth: [array]
    '''
    flux_bins, _bins = np.histogram(time, weights=flux, bins=bins)
    n_bins, _bins = np.histogram(time, bins=bins)
    time_bins = (bins[1:] + bins[:-1])/2.
    flux_smooth = flux_bins/n_bins
    return time_bins, flux_smooth

def broad_tlusty(ipar, pix=0.01, mwv= 4861, R = 2000, pars=None, sps=None, wave=None, laspec_conv=False):
    '''broad tlusty spectrum with a resolution
    parameters:
    ------------------
    ipar [int]
    pix [float] the sample interval of spectra
    mwv [float] the median (or mean) wavelength of spectrum which is used to caculate the gaussian window by resoution
    R [float] spectrum resolution which we want to get
    Tpars [ 2D array] parameters array of TLUSTY grid e.g. np.array([[teff, logg, z, vt]])
    Tlusty_sps [2D array] spectra array of TLUSTY grid
    wave0 [1D array] the original wavelength of TLUSTY spectra
                     e.g. BTLUSTY optical: wave = np.arange(3200.01, 9998.7, 0.01); 
                          OTLUSTY: wave = np.arange(3000, 7500, 0.01)
    wave [1D array] the new wavelength
    returns:
    -------------
    pars [1D array] which has the same parameters with Tpars
    spec [1D array] has same length as wave
    '''
    flux = sps[ipar]
    if laspec_conv:
       if wave is None: wave  =  nnp.arange(3201, 7499, 0.01)
       wv_new, convsp = convolution.conv_spec(wave, flux, R_hi=300000., R_lo=R, over_sample_additional=1,
              gaussian_kernel_sigma_num=5., wave_new=wave,
              wave_new_oversample=1, verbose=False, return_type='array')
    else:
        fwhm = mwv/R
        sigma =  fwhm / (2.0 * np.sqrt(2. * np.log(2.)))
        nn2 = sigma/pix
        x = np.arange(-nn2*2,nn2*2+1,1)
        s2 = stats.norm.pdf(x,0,nn2)
        s = s2/np.sum(s2)
        convsp = np.convolve(flux,s,mode='same')
    return Tpars[ipar], convsp


def normalize_template(ipar, wvl, pars, sps):
    '''Apply rotational broadening to a spectrum. The formulae given in Gray's "The Observation
       and Analysis of Stellar Photospheres". 
       
    parameters:
    ------------------
    ipar [int]
    vsini : [float]
        Projected rotational velocity [km/s].
    epsilon : [float]
        Linear limb-darkening coefficient (0-1).
    wvl : array
        The wavelength array [A]. Note that a
        regularly spaced array is required.
    edgeHandling : string, {"firstlast", "None"}
        The method used to handle edge effects.
    Tpars [ 2D array] parameters array of TLUSTY grid e.g. np.array([[teff, logg, z, vt]])
    Tlusty_sps [2D array] spectra array of TLUSTY grid
    
    returns:
    -------------
    pars [1D array] which has the same parameters with Tpars
    spec [1D array] has same length as wave
    
    '''
    flux = sps[ipar]
    nflux = normalization.normalize_spectrum_spline(wvl, flux)
    return pars[ipar], nflux[0]

def rotbroad_template(ipar, epsilon, vsini, wvl, edgeHandling='firstlast', pars=None, sps=None):
    '''Apply rotational broadening to a spectrum. The formulae given in Gray's "The Observation
       and Analysis of Stellar Photospheres". 
       
    parameters:
    ------------------
    ipar [int]
    vsini : [float]
        Projected rotational velocity [km/s].
    epsilon : [float]
        Linear limb-darkening coefficient (0-1).
    wvl : array
        The wavelength array [A]. Note that a
        regularly spaced array is required.
    edgeHandling : string, {"firstlast", "None"}
        The method used to handle edge effects.
    Tpars [ 2D array] parameters array of TLUSTY grid e.g. np.array([[teff, logg, z, vt]])
    Tlusty_sps [2D array] spectra array of TLUSTY grid
    
    returns:
    -------------
    pars [1D array] which has the same parameters with Tpars
    rflux [1D array] rotational broadening spectra
    
    '''
    flux = sps[ipar]
    if vsini > 0:
       rflux = pyasl.rotBroad(wvl, flux, epsilon, vsini, edgeHandling=edgeHandling)
    else:
       rflux = flux
    return np.append(Tpars[ipar], vsini), rflux

def interpolate_fluxrv(times, phase, flux_phase, tc0, period=None):
    '''interpolate flux or rv by the fluxs at one period
    parameters:
    --------------
    times: [1D array] the times which one want to inteplate
    phase: [1D array] the phase which is given by phoebe
    flux_phase: [1D array] the flux of phase
    tc0: [float] Zeropoint date at superior conjunction (periastron passage) of the primary component 
    period: [period]
    phase0: if phase0 = 0.5 phase: -0.5, 0.5
    returns: 
    -------------
    fluxes: [1 D array]
    '''
    if period is not None:
       tminust0 =times - tc0
       tdividep =  tminust0/period
       _phase = np.mod(tminust0, period)/period
    else:
       _phase = times
    phase = np.append(phase, phase+1)
    flux_phase = np.append(flux_phase, flux_phase)
    fluxes = np.interp(_phase, phase, flux_phase)
    return fluxes


def rvcorr_spec(wave, flux, fluxerr, rv, wave_new=None, left=np.nan, right=np.nan, interp1d=None):
    ''' correct spectrum with radial velocity
    parameters:
    ------------
    wave [1d array]
    flux [1d array]
    fluxerr [1d array]
    barycorr [float] barycentric radial velocity in units of km/s
    
    returns:
    ----------
    flux_bc [1d array]
    fluxerr_bc [1d array]
    '''
    wvl = wave
    flux = flux
    ## Page 71 of An Introduction to Close Binary Stars
    c = 299792.458
    beta = rv/c
    lgwvl = np.log(wvl)
    gamma =(1+beta)/(1-beta)
    _lgwvl = lgwvl + 0.5*np.log(gamma)
    
    if wave_new is None:
       lgwvl = np.log(wvl)
    else: lgwvl = np.log(wave_new)
    if interp1d is None:
       flux_bc = np.interp(lgwvl, _lgwvl, flux, left=left, right=right)
       err2 = np.interp(lgwvl, _lgwvl, fluxerr**2, left=left, right=right)
    else:
       flux_bc = interp1d(_lgwvl, flux, kind='linear',fill_value='extrapolate')(lgwvl)
       err2 = interp1d(_lgwvl, fluxerr**2, kind='linear',fill_value='extrapolate')(lgwvl)
    fluxerr_bc = np.sqrt(err2)
    return flux_bc, fluxerr_bc


from astroquery.vizier import Vizier
from speedysedfit.photometry_query import zpt
from astropy import units
import numpy as np
def get_parallax(objectname, radius=5):

    v_gaia = Vizier(columns=["Plx", "e_Plx", '+_r', 'Gmag', 'nueff', 'pscol', 'ELAT', 'Solved'])

    data = v_gaia.query_object(objectname, catalog=['I/350/gaiaedr3'], radius=radius*units.arcsec)

    if len(data) == 0:
        return None, None

    data = data['I/350/gaiaedr3'][0]

    plx, plx_e = data['Plx'], data['e_Plx']

    if not data['pscol']:
        data['pscol'] = 0

    # apply parallax zero point correction of Lindgren. If not possible, use the average offset.
    # https://arxiv.org/pdf/2012.01742.pdf
    # https://gitlab.com/icc-ub/public/gaiadr3_zeropoint
    try:
        zp = zpt.get_zpt(data['Gmag'], data['nueff'], data['pscol'], data['ELAT'], data['Solved'])
    except Exception:
        warnings.warn("Could not calculate the parallax zero point offset based on Lindgren+2020, using average")
        zp = 0.02
    #plx -= zp

    return plx, plx_e, zp

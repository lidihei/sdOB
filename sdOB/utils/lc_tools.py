import numpy as np
import lightkurve as lk
from scipy.interpolate import CubicSpline
from scipy.stats import bootstrap

def interpbyfoldlc(time, phase, flux, fluxerr, period=None, t0=0,  bc_type='natural'):
    '''
    iterpolate a light curve by using the soomth LC.
    parameters:
    time: [1D array] or the phase of light curve
    phase: [1D array]
    flux: [1D array]
    fluxerr: [1D array]
    period: [float] the period of a light curve, if phase = np.mod((time-t0)/period, 1)
    returns:
    y: [1D array] the flux of the given time
    yerr: [1D array] the flux error of the given time
    '''
    _ind = np.argsort(phase)
    time = time if period is None else np.mod((time-t0)/period, 1)
    phase = phase[_ind]
    flux = flux[_ind]
    fluxerr = fluxerr[_ind]
    phase = np.hstack((phase[:-1]-1, phase, phase[1:]+1))
    flux = np.hstack((flux[:-1], flux, flux[1:]))
    fluxerr2 = np.hstack((fluxerr[:-1], fluxerr, fluxerr[1:]))**2
    func = CubicSpline(phase, flux ,bc_type='periodic')
    funcerr2 = CubicSpline(phase, fluxerr2 ,bc_type='periodic')
    y = func(time)
    yerr = np.sqrt(funcerr2(time))
    return y, yerr

def hist_lc(time, flux, bins):
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

def smooth_by_bin(time, flux, bins, method='median'):
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
    nbins = len(bins)-1
    fluxbin = np.zeros(nbins)
    fluxbinerr = np.zeros(nbins)
    #time_bin = np.zeros(nbins)
    if method.lower() =='median':
       for i in np.arange(nbins):
           _ind = (time > bins[i]) & (time < bins[i+1])
           fluxbin[i] = np.nanmedian(flux[_ind])
           #print('median',fluxbin[i])
           fluxbinerr[i] = 1.253*np.nanstd(flux[_ind])
    else:
       for i in np.arange(nbins):
           _ind = (time > bins[i]) & (time < bins[i+1])
           fluxbin[i] = np.nanmean(flux[_ind])
           #print('mean', fluxbin[i])
           fluxbinerr[i] = np.nanstd(flux[_ind])
    time_bin = (bins[1:] + bins[:-1])/2.
    return time_bin, fluxbin, fluxbinerr

def bin_foldlc(lc, bins, per, tc0, method='median'):
    '''bin the folded light curve
    parameters:
    lc: a light curve of lightkurve
    bins: [array] an array which is used to bin the folded lc
    per: [float] period
    tc0: [float] zero phase time
    returns:
    lcfold: folded light curve
    lcfoldbin
    '''
    lcfold = lc.fold(per, tc0)
    time = lcfold.time; flux = lcfold.flux
    time_bin, fluxbin, fluxbinerr = smooth_by_bin(time, flux, bins, method=method)
    fluxmedian = np.median(fluxbin)
    fluxbin = fluxbin/fluxmedian
    fluxbinerr = fluxbinerr/fluxmedian
    lcfold.flux = lcfold.flux/fluxmedian
    lcfold.flux_err = lcfold.flux_err/fluxmedian
    lcfoldbin = lk.LightCurve(time=time_bin, flux=fluxbin, flux_err=fluxbinerr)
    return lcfold, lcfoldbin


def bin_foldlc1(lc, bins, per, tc0, method='median'):
    '''bin the folded light curve
    parameters:
    lc: a light curve of lightkurve
    bins: [array] an array which is used to bin the folded lc
    per: [float] period
    tc0: [float] zero phase time
    returns:
    lcfold: folded light curve
    lcfoldbin
    fluxmedian: [float]
    '''
    lcfold = lc.fold(per, tc0)
    time = lcfold.time; flux = lcfold.flux
    time_bin, fluxbin, fluxbinerr = smooth_by_bin(time, flux, bins, method=method)
    fluxmedian = np.median(fluxbin)
    fluxbin = fluxbin/fluxmedian
    fluxbinerr = fluxbinerr/fluxmedian
    lcfold.flux = lcfold.flux/fluxmedian
    lcfold.flux_err = lcfold.flux_err/fluxmedian
    lcfoldbin = lk.LightCurve(time=time_bin, flux=fluxbin, flux_err=fluxbinerr)
    return lcfold, lcfoldbin, fluxmedian

def bin_foldlc2(lc, bins, per, tc0, method='median'):
    '''bin the folded light curve
    parameters:
    lc: a light curve of lightkurve
    bins: [array] an array which is used to bin the folded lc
    per: [float] period
    tc0: [float] zero phase time
    returns:
    lcfold: folded light curve
    lcfoldbin
    fluxmedian: [float]
    '''
    lcfold = lc.fold(per, tc0)
    time = lcfold.time; flux = lcfold.flux
    _time = np.hstack([time-1, time, time+1])
    _flux = np.hstack([flux, flux, flux])
    time_bin, fluxbin, fluxbinerr = smooth_by_bin(_time, _flux, bins, method=method)
    fluxmedian = np.median(fluxbin)
    fluxbin = fluxbin/fluxmedian
    fluxbinerr = fluxbinerr/fluxmedian
    lcfold.flux = lcfold.flux/fluxmedian
    lcfold.flux_err = lcfold.flux_err/fluxmedian
    lcfoldbin = lk.LightCurve(time=time_bin, flux=fluxbin, flux_err=fluxbinerr)
    return lcfold, lcfoldbin, fluxmedian



def smooth_bybin_bootstraperr(time, flux, bins, func=np.median, **arg):
    '''smooth light curve by bins. The error is esitmated by bootstrap 
    (details see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
    parameters:
    -------------------
    time: [array] could be phase
    flux: [flux]
    bins: [array]
    func: [function] (e.g. np.mean, np.median)
    returns:
    -------------------
    time_bins: [array] time_bins = (bins[1:] + bins[:-1])/2
    fluxbin: [array]
    fluxbinerr: [array]
    '''
    nbins = len(bins)-1
    fluxbin = np.zeros(nbins)
    fluxbinerr = np.zeros(nbins)
    for i in np.arange(nbins):
        _ind = (time > bins[i]) & (time < bins[i+1])
        fluxbin[i] = func(flux[_ind])
        res = bootstrap((flux[_ind],), func, confidence_level=0.68)
        fluxbinerr[i] = res.standard_error
    time_bin = (bins[1:] + bins[:-1])/2.
    return time_bin, fluxbin, fluxbinerr


def smooth_by_bin2(time, flux, bins, method='median', **arg):
    '''smooth light curve by bins
    parameters:
    -------------------
    time: [array] could be phase
    flux: [flux]
    bins: [array]
    returns:
    -------------------
    time_bins: [array] time_bins = (bins[1:] + bins[:-1])/2
    fluxbin: [array]
    fluxbinerr: [array], if method is mean err = np.std(flux_in_bin)/np.sqrt(number_points_in_bin)
                         if method is median err = 1.253*np.std(flux_in_bin)/np.sqrt(number_points_in_bin)
    '''
    nbins = len(bins)-1
    fluxbin = np.zeros(nbins)
    fluxbinerr = np.zeros(nbins)
    #time_bin = np.zeros(nbins)
    if method.lower() =='median':
       for i in np.arange(nbins):
           _ind = (time > bins[i]) & (time < bins[i+1])
           fluxbin[i] = np.median(flux[_ind])
           #print('median',fluxbin[i])
           fluxbinerr[i] = 1.253*np.std(flux[_ind])/np.sqrt(len(flux[_ind])) # 1.253 = np.sqrt(np.pi/2)
    else:
       for i in np.arange(nbins):
           _ind = (time > bins[i]) & (time < bins[i+1])
           fluxbin[i] = np.mean(flux[_ind])
           #print('mean', fluxbin[i])
           fluxbinerr[i] = np.std(flux[_ind])/np.sqrt(len(flux[_ind]))
    time_bin = (bins[1:] + bins[:-1])/2.
    return time_bin, fluxbin, fluxbinerr


def bin_foldlc3(lc, bins, per, tc0, smoothfunc=smooth_by_bin2, **arg):
    '''bin the folded light curve
    parameters:
    lc: a light curve of lightkurve
    bins: [array] an array which is used to bin the folded lc
    per: [float] period
    tc0: [float] zero phase time
    returns:
    lcfold: folded light curve
    lcfoldbin
    fluxmedian: [float]
    '''
    lcfold = lc.fold(per, tc0)
    time = lcfold.time; flux = lcfold.flux
    _time = np.hstack([time-1, time, time+1])
    _flux = np.hstack([flux, flux, flux])
    time_bin, fluxbin, fluxbinerr = smoothfunc(_time, _flux, bins, **arg)
    fluxmedian = np.median(fluxbin)
    fluxbin = fluxbin/fluxmedian
    fluxbinerr = fluxbinerr/fluxmedian
    lcfold.flux = lcfold.flux/fluxmedian
    lcfold.flux_err = lcfold.flux_err/fluxmedian
    lcfoldbin = lk.LightCurve(time=time_bin, flux=fluxbin, flux_err=fluxbinerr)
    return lcfold, lcfoldbin, fluxmedian

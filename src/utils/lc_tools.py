import numpy as np
import lightkurve as lk

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


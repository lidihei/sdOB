import numpy as np
import matplotlib.pyplot as plt

def lnprob_slam(pars,wave,flux, show=False):
    Xpred = pars
    wvl = np.arange(3700, 7499, 0.5)
    if (pars[0] > np.log10(34000)) or (pars[0] < np.log10(26000)): return -np.inf
    if (pars[1] > 4.25) or (pars[1] < 3.2): return -np.inf
    if (pars[2] > 2) or (pars[2] < 0.1): return -np.inf
    #if (pars[3] > 10.1) or (pars[3] < 0.1): return -np.inf
    if (pars[3] > 250) or (pars[3] < 80): return -np.inf
    y_slam = slmodel.predict_spectra(Xpred)[0]
    y_slam = interp1d(wvl, y_slam, kind='linear',fill_value='extrapolate')(wave)
    dy = np.abs(y_slam-flux)
    _ind = ~np.isnan(dy)
    dy = dy[_ind]
    #print(np.sum(np.isnan(dy)))
    ind = dy<np.percentile(dy,99.5)
    #print(np.sum(ind))
    if show:
      fig, axs = plt.subplots(2,1, figsize=(15, 7), sharex=True, gridspec_kw={'height_ratios': [2,1]})
      plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0)
      e1, = axs[0].plot(wave, flux)
      e2, = axs[0].plot(wave, y_slam)
      axs[1].plot(wave, y_slam-flux, c='grey', alpha=0.5)
      axs[1].scatter(wave[_ind][ind], y_slam[_ind][ind]-flux[_ind][ind], color='b', s=1, alpha=0.5)
      plt.sca(axs[0])
      plt.legend([e1, e2], ['observed', 'slam'])
      plt.ylabel('normalized flux')
      plt.sca(axs[1])
      plt.ylabel('$f_{slam} - f_{obs}$')
      plt.xlabel('wavelength [$\AA$]')
    return -1000*np.sum(dy[ind]**2)


def slam_mcmcfit(w,f,y0):
    # MCMC sampling
    n = len(f)
    
    #start to configure emcee
    nwalkers = 20
    ndim = 4
    p0=np.zeros((nwalkers,ndim))
    p0[:,0] = np.random.rand(nwalkers)*1
    p0[:,1] = np.random.rand(nwalkers)*10
    p0[:,2] = np.random.rand(nwalkers)*1
    p0[:,3] = np.random.rand(nwalkers)*1+y0
    #p0[:,4] = np.random.rand(nwalkers)*0.1+1.
      
    sampler = emcee.EnsembleSampler(nwalkers, \
            ndim, lnprob_sersic, args=[w,f])
    
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    
    sampler.run_mcmc(pos, 10000)
    
    samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
    corner.corner(samples)
    popt = np.median(samples, axis=0)
    pcov = np.zeros((ndim,ndim))
    for i in range(ndim):
        for j in range(ndim):
            pcov[i,j] = (np.sum((samples[:,i]-popt[i])*\
                (samples[:,j]-popt[j])))/len(samples)
    return popt, pcov

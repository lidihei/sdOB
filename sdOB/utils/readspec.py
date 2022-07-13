from astropy.io import fits
import numpy as np


def read_p200_spec(fname):
    ''' read spectrum observed by P2000
    returns:
    ---------------
    wave
    flux
    fluxerr
    '''
    hdulist = fits.open(fname)
    date_keck = hdulist[0].header['UTSHUT']
    flux0=hdulist[0].data
    flux = flux0[0][0]
    temp=hdulist[0].header['CRVAL1']
    step=hdulist[0].header['CD1_1']
    wave = temp + np.arange(hdulist[0].header['NAXIS1'])*step
    fluxerr = flux0[3][0]
    return wave, flux,fluxerr

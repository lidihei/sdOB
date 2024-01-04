from astropy import constants, units
import numpy as np

'''

- $lg(\frac{R}{R\odot}) = 0.73lg(\frac{M}{M\odot})$      $\,\,\,\,\,\,\,\,\,\,\,$ $M >0.4 M\odot$
- $lg(\frac{R}{R\odot}) = lg(\frac{M}{M\odot}) + 0.1$     $\,\,\,\,\,\,\,\,\,\,\,$ $M < 0.4 M\odot$

- $lg(\frac{L}{L\odot}) = 4lg(\frac{M}{M\odot}) + 0.0792$  $\,\,\,\,\,\,\,\,\,\,\,$ $M > M\odot$
- $lg(\frac{L}{L\odot}) = 2.76lg(\frac{M}{M\odot}) - 0.174$  $\,\,\,\,\,\,\,\,\,\,\,$ $M < M\odot$

'''

def M2R(M):
    '''calculate radius from main sequence star mass
    parameter:
    ----------
    M [float] in solar mass
    
    returns:
    ----------
    R [float] stellar radius (Rsun)
    '''
    lgm = np.log10(M)
    if M > 0.4:
       R = 10**(0.73*lgm)
    else:
       R = 10**(lgm + 0.1)
    return R

def M2L(M):
    '''calculate stellar Luminosity from main sequence star mass
    parameter:
    ----------
    M [float] in solar mass
    
    returns:
    ----------
    L [float] stellar Luminosity (Rsun)
    '''
    lgm = np.log10(M)
    if M > 1:
       L = 10**(4*lgm + 0.0792)
    else:
       L = 10**(2.76*lgm-0.174)
    return L

def M2Teff(M):
    '''calculate efficiency temperature from main sequence star mass
    parameter:
    ----------
    M [float] in solar mass
    
    returns:
    ----------
    Teff [float] in units of K
    '''
    R = M2R(M)*units.Rsun
    L = M2L(M)*units.Lsun
    frac = 4*np.pi*R**2*constants.sigma_sb
    Teff = (L/frac)**(0.25)
    return Teff.to('K')


def RT2L(R, T):
    '''calculate log gravity from main sequence star mass and radius
    parameter:
    ----------
    R in solar radius e.g. R = 10*units.Rsun
    T in units of Kelvin e.g T = 50000*units.K
    returns:
    ----------
    L [float] in units of Lsun
    '''
    L =  4*np.pi*(R)**2*constants.sigma_sb*(T)**4
    return L.to('Lsun')

def LT2R(L, T):
    '''calculate stellar radius by luminosity and temperature
    parameters:
    -------------
    L [float] stellar luminostiy
    T [float] temperature
    returns:
    ------------
    R [float] in solar radius
    '''
    L = L * units.Lsun
    T = T * units.K
    sigma = constants.sigma_sb
    R = np.sqrt(L/(4*np.pi*sigma*T**4))
    return R.to('Rsun').value

def LR2T(L, R):
    '''
    parameters:
    L [float]  in Lsun
    R [float] in  Rsun
    returns:
    Teff [float] in K
    '''
    L = L * units.Lsun
    sigma = constants.sigma_sb
    R = R*units.Rsun
    Teff = (L/(4*np.pi*sigma*R**2))**(0.25)
    return Teff.to('K').value


def Rlogg2M(R, logg):
    '''calculate stellar mass by stellar radius and logg
    parameters:
    -----------
    R:[float] stellar radius
    logg: [float] effective gravity
    
    returns:
    ---------
    M: stellar mass
    '''
    R = R * units.Rsun
    g = 10**logg * units.cm/units.s**2
    M = g* R**2/constants.G
    return M.to('Msun').value

def Mlogg2R(M, logg):
    '''calculate stellar mass by stellar radius and logg
    parameters:
    -----------
    M: [float] stellar mass e.g. 1
    logg: [float] effective gravity e.g. 4.5
    
    returns:
    ---------
    R:[float] stellar radius
    '''
    M = M * units.Msun
    g = 10**logg * units.cm/units.s**2
    R = np.sqrt(M*constants.G/g)
    return R.to('Rsun').value

def MR2logg(M, R):
    '''calculate log gravity from main sequence star mass and radius
    parameter:
    ----------
    M [float] in solar mass
    R [float] in solar radius
    returns:
    ----------
    logg [float] log(cm/s2)
    '''
    R = R*units.Rsun
    M = M*units.Msun
    g = constants.G * M/R**2
    logg = np.log10(g.to('cm/s2').value)
    return logg



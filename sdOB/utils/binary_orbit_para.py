from astropy import constants, units
import numpy as np
from scipy import optimize
from .math import cubicequation_solution


'''
1. [orbital period](https://en.wikipedia.org/wiki/Semi-major_and_semi-minor_axes)
2. [SPECTROSCOPIC BINARY STARS](https://phys.libretexts.org/Bookshelves/Astronomy__Cosmology/Book%3A_Celestial_Mechanics_(Tatum)/18%3A_Spectroscopic_Binary_Stars/18.01%3A_Introduction_to_Spectroscopic_Binary_Stars)

$n = 2\pi/P$

$K_1 = \frac{na_1\sin i}{\sqrt{1-e^2}}$

$n^2 a_{1}^3 = GM$


$M = m_{2}^3/(m_1+m_2)^2$

- $n = \frac{2\pi}{P}$
- $a_1^3 n^2 = G \frac{m_2^3}{(m_1+m_2)^2}$
- $a_2^3 n^2 = G \frac{m_1^3}{(m_1+m_2)^2}$
- $a = a_1 + a_2$
- $a^3 n^2 = G(m_1 + m_2)$

'''

def pk2asini(P, K, e=0):
    '''calculate projected semi-major axis of orbit by using semi-amplitude of rv, period and eccentric
    parameters
    ----------
    P [time units] orbital period e.g. 1*units.day
    K [velocity units] e.g. 10 * units.km/units.s
    e [float] eccentric
    return
    -------
    asini projected semi-major axis of orbit
    '''
    n = 2*np.pi/P
    asini = K*np.sqrt(1-e**2)/n
    return asini

def pm1k1qe2sini(P, m1, K1, q, e):
    '''calculate sini by using orbtial period, q(m2/m1), k1, m1 and e
    parameters:
    -----------------------
    P [time units] orbital period e.g. 1*units.day
    K1 [velocity units] e.g. 10 * units.km/units.s
    m1 [float] stellar mass e.g. 1*units.Msun
    q [float] mass ratio (q = m2/m1)
    e [float] eccentric
    returns:
    ----------
    sini
    '''
    gnm = constants.G*(2*np.pi/P*m1)
    K0 = gnm**(1/3)*q/(1+q)**(2./3)/np.sqrt(1-e**2)
    sini = K1/K0.to('km/s').value
    return sini

def period2M(P, K1, i=90, e = 0):
    '''calculate M by using period and amplitude of raidal velocity
    parameters
    -----------
    P [time units] period e.g. 1*units.day
    K [velocity units] e.g. 10 * units.km/units.s
    i [degree]inclination angle
    e [float] eccentric
    '''
    n = 2*np.pi/P
    a1 = K1*np.sqrt(1-e**2)/(n*np.sin(np.deg2rad(i)))
    M = n**2 * a1**3/constants.G
    return M, a1

def M2m2_newton(M, m1, maxiter=2000, x1=None):
    ''' calculate m2 by using M and m1 with Newton iteration method
    parameters
    ------------
    M [float] the total mass (m1+m2) e.g. 1
    m1 [float] e.g. 1
    maxiter [int] the iteration of Newton method
    x1: [float] see scipy.optimize.newton
    return
    ------
    m2 [float] mass of the secondary
    '''
    def f(x, M, m1):
        y = x**3 - M*x**2 - 2*M*m1*x - M*m1**2
        return y
    if x1 is None:
       try:
           m2 = optimize.newton(f, 0,x1=m1, maxiter=maxiter, args=(M, m1))
       except:
           m2 = optimize.newton(f, 0,x1=2*m1, maxiter=maxiter, args=(M, m1))
    else:
       m2 = optimize.newton(f, 0,x1=x1, maxiter=maxiter, args=(M, m1))
    return m2

def M2m2(M, m1):
    ''' calculate m2 by using M and m1 with cubic equation solution
    parameters
    ------------
    M [float] the total mass (m1+m2) e.g. 1
    m1 [float] e.g. 1
    maxiter [int] the iteration of Newton method
    x1: [float] see scipy.optimize.newton
    return
    ------
    m2 [float] mass of the secondary
    '''
    def f(x, M, m1):
        y = x**3 - M*x**2 - 2*M*m1*x - M*m1**2
        return y
    a = 1; b = -M; c = -2*M*m1; d = - M*m1**2
    xs = np.array(cubicequation_solution(a, b, c, d))
    _ind = np.isreal(xs) & (xs>0)
    m2 = np.real(xs[_ind])
    return m2


def m1pkie2m2qa_newton(m1, P, K1, i, e, x1=None):
    '''calculate mass ratio by using primary mass, obtial period and semi-amplitude of raidal velocity ( Newton iteration method)
    parameters
    -----------
    m1 [float] in solar mass e.g. 1
    P [time units] period e.g. 1*units.day
    K [velocity units] e.g. 10 * units.km/units.s
    i [degree]inclination angle
    e [float] eccentric
    x1: [float] see scipy.optimize.newton
    returns
    ------
    m2 [float] the secondary mass (in solar mass)
    q [float] mass ratio (m2/m1)
    a [distance units] semi-major axis of the binary system
    '''
    M, a1 = period2M(P, K1, i=i, e = e)
    M = M.to('Msun').value
    m2 = M2m2_newton(M, m1, maxiter=2000, x1=x1)
    q = m2/m1
    a2 = a1/q
    a = a1+ a2
    return m2, q, a.to('Rsun')


def m1pkie2m2qa(m1, P, K1, i, e):
    '''calculate mass ratio by using primary mass, obtial period and semi-amplitude of raidal velocity (cubic equation solution)
    parameters
    -----------
    m1 [float] in solar mass e.g. 1
    P [time units] period e.g. 1*units.day
    K [velocity units] e.g. 10 * units.km/units.s
    i [degree]inclination angle
    e [float] eccentric
    x1: [float] see scipy.optimize.newton
    returns
    ------
    m2 [float] the secondary mass (in solar mass)
    q [float] mass ratio (m2/m1)
    a [distance units] semi-major axis of the binary system
    '''
    M, a1 = period2M(P, K1, i=i, e = e)
    M = M.to('Msun').value
    m2 = M2m2(M, m1)
    q = m2/m1
    a2 = a1/q
    a = a1+ a2
    return m2, q, a.to('Rsun')


def PRincl2vsini(period, R, incl=None, sini=None):
    ''' calculate vsini by using period, stellar radius and inclination.
    parameters:
    --------------
    R: [Rsun] e.g. 0.9
    period: [day] e.g. 1.5 
    incl: [deg]
    sini: [float]  [0, 1]
    returns:
    ------------
    vrotation: [float] in km/s
    vsini: [float] in km/s e.g 77
    '''
    if sini is None: sini = np.sin(np.rad2deg(incl))
    vrotation = 50.592732 * R/period
    vsini = vrotation*sini
    return vrotation, vsini

def RPvsini2incl(R, period, vsini):
    '''asumming synchronized, calculate inclination angle by radius of star(R), period of orbital and projected rotation velocity (vsini)
    parameters:
    --------------
    R: [Rsun] e.g. 0.9
    period: [day] e.g. 1.5 
    vsini: [km/s] e.g 77
    returns:
    ------------
    incl: [deg]
    sini: [float]  [0, 1]
    '''
    R = R*units.Rsun
    period = period*units.day
    vsini = vsini*units.km/units.s
    v = 2*np.pi*R/period
    v = v.to('km/s').value
    sini = vsini/v
    incl = np.arcsin(sini)
    incl = np.rad2deg(incl)
    return incl, sini

def m1qperiod2sma(m1, q, period):
    '''calculate semi-major axis of the binary system by m1, mass ration (q = m2/m1) and obtial period. 𝑎3𝑛2=𝐺(𝑚1+𝑚2)
    parameters
    -----------
    m1 [mass unit] in solar mass e.g. 1*units.Msun
    q [float] mass ratio (m2/m1)
    period [time units] period e.g. 1*units.day
    return 
    ------
    sma [distance units] semi-major axis of the binary system
    '''
    M = m1 + m1*q
    n2 = (2*np.pi/period)**2
    sma = (constants.G * M/n2)**(1./3.)
    return sma


def Msma2period(M, sma):
    '''calculate orbital period by total mass and semi-major axis
    parameters
    -----------
    M [mass unit] the total mass of binary M = m1+m2,  e.g. 1*units.Msun
    sma [distance units] semi-major axis of the binary system, e.g. 10*units.Rsun
    return 
    period [time units] period
    ------
    '''
    n = np.sqrt(constants.G * M/sma3)
    period = 2*np.pi/n
    return period


def binarymassfunc(P, K1):
    '''$f = \frac{M^3_2\sin^3i}{(M_1+M_2)^2} = \frac{P_{\rm orb} K_1^3}{2\pi G}$
    parameters:
    --------------
    P: [time units] period e.g. 1*units.day
    K1: [velocity units] e.g. 10 * units.km/units.s
    returns:
    --------------
    f: [mass units (Msun)] the binary mass function
    '''
    f = P*K1**3/2/np.pi/constants.G
    return f.to('Msun')


def m1bftom2(m1, bf, i=90, maxiter=2000, x1=None):
    ''' calculate m2 by using m1 and binary mass functon
    parameters:
    ---------------
    m1: [float] in units of Msun e.g. m1 =1, the mass of star1
    bf: [float] in units of Msun e.g. bf =1, the binary mass function
    i: [float] inclination of orbital  in units of degree
    maxiter [int] the iteration of Newton method
    x1: [float] see scipy.optimize.newton
    returns:
    --------------
    m2: [float] in units of Msun
    '''
    sini3 = np.sin(np.deg2rad(i))**3
    fsini = bf/sini3
    def func(x, fsini, m1):
        y = x**3 - fsini*(m1+x)**2
        return y
    if x1 is  None:
       try:
           m2 = optimize.newton(func, 0,x1=m1, maxiter=maxiter, args=(fsini, m1))
       except:
           m2 = optimize.newton(func, 0,x1=2*m1, maxiter=maxiter, args=(fsini, m1))
    else:
       m2 = optimize.newton(func, 0,x1=x1, maxiter=maxiter, args=(fsini, m1))
    return m2

def eR_Rochelobe(q):
    '''q is mass ratio (r_L1 : m1/m2)
    ## Equation (2) of Peter P. Eggleton 1983 ApJ
    parameters:
    ------------------
    q [float]
    returns:
    ------------------
    r_L [float], r_L = R_L/sma
    '''
    q23 = q**(2./3.)
    q13 = q**(1./3.)
    r_L = 0.49*q23/(0.6*q23 + np.log(1+q13))
    return r_L

def rsmaq2fillout_factor(requiv, sma, q, F=1, d=1, s=np.array([0., 0., 1.]), component=1 ):
    '''
    parameters:
    -----------
    * `requiv` (float): the equivalent radius
    * `sma` (float): the semi-major axis of the orbit
    * `q` (float): the mass-ratio in the frame of `component`.  If necessary,
        call <phoebe.distortions.roche.q_for_component> first.
    * `F` (float): the synchronicity parameter.
    * `d` (float): instantaneous separation between the two components in the
        orbit.
    * `s` (array of length 3): rotation spin vector, in roche coordinates, at
        the requested time.
    * `component` (int, optional, default=1): whether the requested component
        is the primary (1) or secondary (2) component. 
    returns:
    pot (float): the equipotential in the primary frame
    FF (float): the fillout factor; FF = 0 corresponds to the semi-detached case, FF < 0 to the detached case, and FF ∈ (0, 1) to the contact case.
    '''
    from phoebe.distortions.roche import requiv_to_pot
    from phoebe.constraints.builtin import pot_to_fillout_factor
    pot = requiv_to_pot(requiv, sma, q, F, d, s=s, component=component)
    FF = pot_to_fillout_factor(q, pot)
    return pot, FF

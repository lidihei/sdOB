'''calculate mass loss rate by using bow shock nebula (Kobulnicky et al. 2017)
'''

def vwind(M, R, a, beta=0.8, Gamma=0, alpha = 2.6):
    '''calculate wind veloctiy of O type star
    wind velocity ([Sen et al. 2021](https://ui.adsabs.harvard.edu/abs/2021arXiv210601395S/abstract))
    parameters:
    M: the stellar mass e.g. M = 10 * units.Msun
    R: the stellar radius e.g. R = 5*units.Rsun
    a: the obital separation e.g. a= a*units.Rsun
    beta: the value of bata is 0.8-1 for O stars (Groenewegen & Lamers 1989; Puls et al. 1996)
    Gamma: the Eddington factor
    alpha: [float] the terminal velocity v_inf = alpha*v_esc (escape velocity of star)
    returns:
    v_wind: the velocity of wind
    '''
    v_esc = np.sqrt(2*constants.G*M*(1-Gamma)/R)
    v_inf = alpha *v_esc #(Vink et al. 2001)
    #print(f'v_esc={v_esc.to("km/s")}')
    #v_inf = v_esc #(Vink et al. 2001)
    #print(v_inf.to('km/s'))
    v_wind = v_inf*(1-R/a)**beta
    return v_wind.to('km/s')

class DL07spec():
    
    def __init__(self, model, dire):
        '''
        # Draine & Li 2007 [DL07spec](https://ui.adsabs.harvard.edu/abs/2007ApJ...657..810D/abstract)
        parameters:
        ---------------
        model: [str], e.g. 'MW3.1_00', 'MW3.1_10', 'MW3.1_20', ' MW3.1_30', 'MW3.1_40'
                      'MW3.1_50', 'MW3.1_60', 'LMC2_00', 'LMC2_05','LMC2_10', 'SMC'
        dire: [stri], the directory of DL07spce table, e.g. 'home/lijiao/lijiao/Documents/DL07spec'
        '''
        Ustri = ['0.10', '0.15', '0.20', '0.30', '0.40', '0.50', '0.70', '0.80', '1.00',
                 '1.20', '1.50', '2.00', '2.50', '3.00', '4.00', '5.00', '7.00', '8.00',
                 '12.0', '15.0', '20.0', '25.0', '1e2', '3e2', '1e3', '3e3', '1e4', '3e4', '1e5', '3e5']
        U = [np.float(_) for _ in Ustri]
        self.Ustri = Ustri
        self.U = U
        self.model = model
        self.dire = dire
        
    
    def getgrid(self, filtername=None):
        '''
        returns:
        Ugrid: [array]
        lamgrid: [array]
        Jnvgrid: [array]
        '''
        model = self.model
        Ustris = self.Ustri
        dire = self.dire
        Ugrid, lamgrid, Jnugrid = [], [], []
        for i, Ustri in enumerate(Ustris):
            fname = os.path.join(dire, f'U{Ustri}', f'U{Ustri}_{Ustri}_{model}.txt')
            lam, nudpdu, Jnu = self.readU(fname)
            if filtername is None:
               _U = self.U[i]*np.ones(len(lam))
               Ugrid.append(_U)
               lamgrid.append(lam)
               Jnugrid.append(Jnu)
            else:
               filterlam, Trans = self.readfile(filtername)
               Trans = np.interp(lam, filterlam, Trans, left=0., right=0.)
               Jnuband = np.sum(Jnu*Trans)/np.sum(Trans)
               Ugrid.append(self.U[i])
               Jnugrid.append(Jnuband)
        Ugrid = np.array(Ugrid).ravel()
        lamgrid = np.array(lamgrid).ravel() 
        Jnvgrid = np.array(Jnugrid).ravel()
        self.Ugrid, self.lamgrid, self.Jnugrid = Ugrid, lamgrid, Jnvgrid
        return Ugrid, lamgrid, Jnvgrid
        
        
    def interpolateJnu(self, U, lam=None, filtername=None, log=True, **keywords):
        '''
        parameters:
        -------------
        U: [float]
        lam: [float] the wavelength (um)
        filtername: [stri]
        returns:
        -------------
        Jnv [float] (Jy cm2 sr-1 H-1)
        '''
        Ugrid, lamgrid, Jnvgrid = self.getgrid(filtername=filtername)
        if log:
           Ugrid = np.log10(Ugrid)
           U = np.log10(U)
        if filtername is not None:
           Jnv = np.interp(U, Ugrid, Jnvgrid)
        else:
           grids = np.zeros((len(lamgrid), 2))
           grids[:, 0 ] = Ugrid
           grids[:, 1 ] = lamgrid
           Jnv = griddata(grids, Jnugrid, (U, lam),  **keywords)
        return Jnv
        
        
    def readU(self, fname):
        '''
        returns:
        ---------
        lam: [array] wavelength, in units of um
        nudpdu: [array] nu*dP/dnu
        jnu: [arrya] (Jy cm2 sr-1 H-1)
        '''
        f = open(fname, 'r')
        lines = f.readlines()
        for _i, line in enumerate(lines):
            if 'lambda    nu*dP/dnu     j_nu' in line:
               break
        skiprows = _i+2
        import pandas as pd
        names=['lam', 'nudpdnu', 'jnu']
        Udata = pd.read_table(fname, skiprows=skiprows,sep='\s+', names = names)
        _ind = np.argsort(Udata.lam.values)
        lam = Udata.lam.values[_ind]
        nudpdnu = Udata.nudpdnu.values[_ind]
        jnv = Udata.jnu.values[_ind]
        return lam, nudpdnu, jnv
    
    def readfile(self, filtername):
        import pandas as pd
        ''' filter data downloaded from http://svo2.cab.inta-csic.es/svo/theory/fps3/
        filtername: [stri] e.g. '/home/lijiao/lijiao/Documents/filters/WISE/WISE.W4.dat'
        returns:
        lam: [array] wavelength, in units of um
        Trans: [array] Transmission
        '''
        filterdata = pd.read_table(filtername, skiprows=5,sep='\s+', names = ['lam', 'Trans'])
        lam = filterdata.lam.values/1e4
        Trans = filterdata.Trans.values
        return lam, Trans
    
    

def RDP_U(Rstar, Teff, Ro):
  '''Radiation density parameter, eqution (1) of [Kobulnicky et al. 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..201K/abstract) 
  which is estimated by Mathisetal. (1983)
  parameters:
  Rstar: radius of star e.g. 3*units.Rsun
  Teff: effective temperature of star, e.g. 20000*units.K
  Ro: the standoff distance between star and nebula, e.g. 0.1*units.pc
  returns:
  ---------------
  U: [float] dimesionless ratio
  '''
  from astropy import units, constants
  Ustar = constants.sigma_sb * Rstar**2*Teff**4/Ro**2
  Ustar = Ustar.to('erg/s/cm2').value
  U = Ustar/0.0217
  return U

def mdot(Ro_deg, D, Inv, Vw, lr_deg, Jnv, Va=None, mmu =None, ):
    ''' Eq. (7) of [Kobulnicky et al. 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...856...74K/abstract)
    Ro_deg: the standoff distance between star and nebula, (arcsec, e.g = 18*units.arcsec)
    #D: is the distance of the bow shock stars e.g. 1.units.kpc
    Inu: is the specific intensity at a selected infrared frequency Jy sr−1
    Vw: is the wind veloctiy of e.g. 2380*units.km/units/s
    lr_deg: is the path length through the dust nebula, (arcsec, e.g. 40*units.arcsec)
    Jnv: The dust emission coefficient per nucleon, jν,inJycm2sr−1 nucleon−1 as tabulated by Draine&Li (2007)
    Va: is the velocity of the ambient interstellar medium (ISM) e.g. 30*units.km/units.s
    mmu: is the mean particle mass e.g. 2.3e-24*units.g
    '''
    Rorad = Ro_deg.to('rad').value
    lrrad =lr_deg.to('rad').value
    if Va is None: Va = 30*units.km/units.s #a typical “runaway” speed of bow shock stars
    if mmu is None: mmu = 2.3e-24*units.g #(appropriate to the Milky Way ISM)
    numerator = np.pi*mmu*Rorad**2*D*Va**2*Inv
    dinominator = Vw*lrrad*Jnv
    mdot = numerator/dinominator
    return mdot.to('Msun/yr')

if __name__ =='__main__':
   size = 1000
   Rodeg = np.random.normal(24.6, 12, size=size)*units.arcsec
   D = 1/np.random.normal(0.362, 0.02, size=size)*units.kpc
   Ro = D * Rodeg.to('rad').value
   lrdeg =np.random.normal(40, 12, size=size)*units.arcsec
   lr =D * lrdeg.to('rad').value
   Rstar = np.random.normal(9.1, 0.5, size=size)*units.Rsun
   Teff = 30681*units.K
   M = np.random.normal(22.5, 3, size=size)*units.Msun
   A = np.random.normal(30.6, 1.5, size=size)*units.Rsun
   Vw = vwind(M, Rstar, A, beta=0.8, Gamma=0)
   Inv = 15474424.10227796*units.Jy
   
   U = RDP_U(Rstar, Teff, Ro)
   
   
   models = ['MW3.1_00', 'MW3.1_10', 'MW3.1_20', 'MW3.1_30', 'MW3.1_40', 'MW3.1_50', 'MW3.1_60']
   dire = '/home/lijiao/lijiao/Documents/DL07spec'
   filtername = '/home/lijiao/lijiao/Documents/filters/WISE/WISE.W4.dat'
   mdots = []
   for model in models:
       DL07_spec = DL07spec(model, dire)
       Jnv = DL07_spec.interpolateJnu(U, lam=None, filtername=filtername, log=True)
       Jnv = Jnv * units.Jy*units.cm**2
       _mdots = mdot(Rodeg, D, Inv, Vw, lrdeg, Jnv, Va=None, mmu =None,)
       mdots.append(_mdots.value)
   
   mdots = np.array(mdots)

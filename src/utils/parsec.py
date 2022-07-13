from scipy.interpolate import griddata
import numpy as np
from astropy.table import Table
from astropy import units, constants

def parsecMZ2LgR(L, MH, logLMH2mass=None, logLMH2logTe=None):
    '''
    parameters:
    -----------
    L: [float]
    MH: []
    returs:
    mass: [array]
    Teff: [array]
    logg: [array]
    R: [array]
    '''
    import xgboost as xgb
    if type(L) !=  np.ndarray: L = np.array([L])
    logL = np.log10(L)
    inputarray = np.ones((logL.shape[0], 2))
    inputarray[:, 0] = logL 
    inputarray[:, 1] = MH
    paras = xgb.DMatrix(data = inputarray)
    mass = logLMH2mass.predict(paras)
    logTe = logLMH2logTe.predict(paras)
    Teff = 10**logTe
    R = np.sqrt(L*units.Lsun/ (4*np.pi *constants.sigma_sb * (Teff*units.K)**4))
    g = constants.G*mass*units.Msun/R**2
    g = g.to('cm/s2').value
    return mass, Teff, np.log10(g), R.to("Rsun").value

class xgb_parsecLMH():

      def __init__(self, xgbmodeldic):
          self.model_dic = xgbmodeldic

      def MTgR(self, L, MH, logLMH2mass=None, logLMH2logTe=None):
          model_dic = self.model_dic
          if logLMH2mass is None: logLMH2mass = model_dic['logLMH2mass']
          if logLMH2logTe is None: logLMH2logTe = model_dic['logLMH2logTe']
          return parsecMZ2LgR(L, MH, logLMH2mass=logLMH2mass, logLMH2logTe=logLMH2logTe)

#ind  = (_tab['label']>0)&(_tab['label']<2) & (_tab['Zini'] >  0.001) & (_tab['logg'] > 3.7) & (_tab['Mini'] > 1) & (_tab['Mini'] < 25) & (_tab['logAge'] < 7)
# parsecgrid = Table.read('/share/HDD6/jdli/data/star_model/PARSEC_logage_6to10_MH_n2p5_0p6.csv')
class parsecLMH():
    
    def __init__(self, parsecgrid, ind):
        '''
        parsecgrid [staropy.table] Table.read('/share/HDD6/jdli/data/star_model/PARSEC_logage_6to10_MH_n2p5_0p6.csv')
        ind [bool array] select a proper range to interpolate
        '''
        parsecgrid = parsecgrid[ind]
        pointsgrid = np.zeros((len(parsecgrid), 2))
        pointsgrid[:,0] = parsecgrid['logL']
        pointsgrid[:,1] = parsecgrid['MH']
        self.gridmass = parsecgrid['Mass']
        self.gridlogTe =  parsecgrid['logTe']
        self.pointsgrid = pointsgrid
    
    def MTgR(self, L, MH, **keywords):
        '''
        returns:
        ----------
        mass
        Teff
        logg
        R
        '''
        if type(L) !=  np.ndarray: L = np.array([L])
        logL = np.log10(L)
        points = np.ones((logL.shape[0], 2))
        points[:,0] = logL
        points[:,1] = MH
        mass = griddata(self.pointsgrid, self.gridmass, points,  **keywords)
        logTe = griddata(self.pointsgrid, self.gridlogTe, points,  **keywords)
        Teff = 10**logTe
        R = np.sqrt(L*units.Lsun/ (4*np.pi *constants.sigma_sb * (Teff*units.K)**4))
        g = constants.G*mass*units.Msun/R**2
        g = g.to('cm/s2').value
        return mass, Teff, np.log10(g), R.to("Rsun").value

    def M(self, L, MH, **keywords):
        '''
        returns:
        ----------
        mass
        Teff
        logg
        R
        '''
        if type(L) !=  np.ndarray: L = np.array([L])
        logL = np.log10(L)
        points = np.ones((logL.shape[0], 2))
        points[:,0] = logL
        points[:,1] = MH
        mass = griddata(self.pointsgrid, self.gridmass, points,  **keywords)
        return mass

if __name__ == '__main__':
   parsecgrid = Table.read('/share/HDD6/jdli/data/star_model/PARSEC_logage_6to10_MH_n2p5_0p6.csv')
   ind  = (parsecgrid['label']>0)&(parsecgrid['label']<2) & (parsecgrid['Zini'] >  0.001) & (parsecgrid['logg'] > 3.7) & (parsecgrid['Mini'] > 1) & (parsecgrid['Mini'] < 25) & (parsecgrid['logAge'] < 7)
   parseclmh = parsecLMH(parsecgrid, ind)
   _m, _, _, _ = parseclmh.MTgR(2000, np.log10(0.38))

   massMH2logLlogg = joblib.load('../data/joblibdump/isochrone/massMH2logLlogg.dump')
   m_BO2s, Teff_BO2s, logg_BO2s, R_BO2s = parsecMZ2LgR(2000, np.log10(0.38), logLMH2mass=massMH2logLlogg['logLMH2mass'], logLMH2logTe=massMH2logLlogg['logLMH2logTe'])

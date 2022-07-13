'''
build a grid of  The coefficients for a 4-parameter limb-darkening law and the gravity
darkening coefficient, y, from the tables provided by Claret
and Bloemen [(2011A&A...529A..75C)](http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/529/A75)
and Claret 2017 [(2017A&A...600A..30C)](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/600/A30) for ATLAS stellar atmosphere models and an
assumed micro-turbulent velocity xi=2 km/s calculated using a least-squares fit.
'''

import xgboost as xgb
import numpy as np
from scipy.interpolate import griddata


def xgb_lyd(logTeff, logg, logZ, xi, model_dic = None):
    '''using xgboost model interpolate  a 4-parameter limb-darkening law (a1, a2, a3, a4) 
       and the gravity darkening coefficient (y).
    model_dic: [a distionary] contains models produced by xgboost
    returns:
    ---------------------
    a1, a2, a3, a4, y
    '''
    paras = xgb.DMatrix(data= np.array([[logTeff, logg, logZ, xi]]))
    a1 = model_dic['model_a1'].predict(paras)[0]
    a2 = model_dic['model_a2'].predict(paras)[0]
    a3 = model_dic['model_a3'].predict(paras)[0]
    a4 = model_dic['model_a4'].predict(paras)[0]
    y = model_dic['model_y'].predict(paras)[0]
    return a1, a2, a3, a4, y


class xgb_ldyClaret():
      def __init__(self, model_dic):
          self.model_dic = model_dic

      def interpolate_lyd(self, logTe, logg, logZ, xi):
          model_dic = self.model_dic
          return xgb_lyd(logTe, logg, logZ, xi, model_dic =model_dic)

from scipy.interpolate import griddata

class lydClaret():

    def __init__(self, grid_eq4, grid_y):
        ''' coefficients for a 4-parameter limb-darkening law and the gravity 
        darkening coefficient, y, from the tables provided by Claret
        and Bloemen [(2011A&A...529A..75C)](http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/529/A75)
        and Claret 2017 [(2017A&A...600A..30C)](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/600/A30) for ATLAS stellar atmosphere models
        (a1 = eLSM, a2 = fLSM, a3=ePCM, a4=fPCM)
        parameters:
        -------------
        grid_eq4: [FITS_rec], the grid of 4-parameter limb-darkening law, e.g. grideq4 = fits.getdata('J_A+A_600_A30_table27.dat.gz.fits')
        grid_y: [FITS_rec], the grid of the gravity darkening coefficient, e,g, grid_y = fits.getdata('J_A+A_600_A30_table29.dat.gz.fits')
        '''
        pointseq4 = np.zeros((len(grid_eq4), 4))
        pointseq4[:, 0] = np.log10(grid_eq4['Teff'])
        pointseq4[:, 1] = grid_eq4['logg']
        pointseq4[:, 2] = grid_eq4['Z']
        pointseq4[:, 3] = grid_eq4['xi']
        self.grida1 = grid_eq4['eLSM']
        self.grida2 = grid_eq4['fLSM']
        self.grida3 = grid_eq4['ePCM']
        self.grida4 = grid_eq4['fPCM']
        self.pointseq4 = pointseq4
        pointsy = np.zeros((len(grid_y), 4))
        pointsy[:, 0] = grid_y['logTeff']
        pointsy[:, 1] = grid_y['logg']
        pointsy[:, 2] = grid_y['Z']
        pointsy[:, 3] = grid_y['xi']
        self.pointsy = pointsy
        self.gridy = grid_y['y']

    def interpolate_lyd(self, logTe, logg, Z, Xi, **keywords):
        '''interpolate coefficients for a 4-parameter limb-darkening law (Claret 2017)
        (a1 = eLSM, a2 = fLSM, a3=ePCM, a4=fPCM)
        parameters:
        -------------
        logTe, logg, Z, Xi
        returns:
        a1, a2, a3, a4
        '''
        if type(logTe) !=  np.ndarray: logTe = np.array([logTe])
        points = np.ones((logTe.shape[0], 4)) 
        points[:,0] = logTe
        points[:,1] = logg
        points[:,2] = Z
        points[:,3] = Xi
        points = (logTe, logg, Z, Xi)
        a1 = griddata(self.pointseq4, self.grida1, points,  **keywords)
        a2 = griddata(self.pointseq4, self.grida2, points,  **keywords)
        a3 = griddata(self.pointseq4, self.grida3, points,  **keywords)
        a4 = griddata(self.pointseq4, self.grida4, points,  **keywords)
        y = griddata(self.pointsy, self.gridy, points,  **keywords)
        return a1, a2, a3, a4, y

    def interpolate_lyd2(self, logTe, logg, Z, **keywords):
        '''interpolate coefficients for a 4-parameter limb-darkening law (Claret 2017)
        (a1 = eLSM, a2 = fLSM, a3=ePCM, a4=fPCM), Xi =2
        parameters:
        -------------
        logTe, logg, Z
        returns:
        a1, a2, a3, a4
        '''
        if type(logTe) !=  np.ndarray: logTe = np.array([logTe])
        points = np.ones((logTe.shape[0], 3)) 
        points[:,0] = logTe
        points[:,1] = logg
        points[:,2] = Z
        pointseq4 = self.pointseq4[:, :3]
        pointsy = self.pointsy[:, :3]
        a1 = griddata(pointseq4, self.grida1, points,  **keywords)
        a2 = griddata(pointseq4, self.grida2, points,  **keywords)
        a3 = griddata(pointseq4, self.grida3, points,  **keywords)
        a4 = griddata(pointseq4, self.grida4, points,  **keywords)
        y  = griddata(pointsy, self.gridy, points,  **keywords)
        return a1, a2, a3, a4, y


if __name__ == '__main__':
  dire = '/home/lijiao/lijiao/limbdarkening/Claret'
  fname_eq4 = os.path.join(dire, 'J_A+A_600_A30_table27.dat.gz.fits')
  fname_y = os.path.join(dire, 'J_A+A_600_A30_table29.dat.gz.fits')

  hdulist_eq4 = fits.open(fname_eq4)
  data = hdulist_eq4[1].data
  hdulist_eq4.close()

  hdulist_y = fits.open(fname_y)
  data_y = hdulist_y[1].data
  hdulist_y.close()

  _data = data[(data['Teff'] > 10000) & (data['Teff'] < 35000) & (data['logg'] < 5) & (data['logg'] > 3.5)]
  _data_y = data_y[(data_y['logTeff'] > 4) & (data_y['logTeff'] < np.log10(35000)) & (data_y['logg'] < 5) & (data_y['logg'] > 3.5)]
  lydd = lydClaret(_data, _data_y)
  
  a1, a2, a3, a4, y = lydd.interpolate_lyd(np.log10(30000), 4, 0, 2)


  ### interpolate with 3 parameters (logTe, logg, Z)
  _data = data[(data['Teff'] > 10000) & (data['Teff'] < 35000) & (data['logg'] < 5) & (data['logg'] > 3.5) & (data['Xi'] ==2)]
  _data_y = data_y[(data_y['logTeff'] > 4) & (data_y['logTeff'] < np.log10(35000)) & (data_y['logg'] < 5) & (data_y['logg'] > 3.5)& (data['Xi'] ==2)]
  lydd = lydClaret(_data, _data_y)
  a1, a2, a3, a4, y = lydd.interpolate_lyd2(np.log10(30000), 4, 0)


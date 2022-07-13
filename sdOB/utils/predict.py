# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed

from scipy.optimize import leastsq, least_squares
import ellc
#from .lyd import lydClaret
#
#dire = '/home/lijiao/lijiao/limbdarkening/Claret'
#fname_eq4 = os.path.join(dire, 'J_A+A_600_A30_table27.dat.gz.fits')
#fname_y = os.path.join(dire, 'J_A+A_600_A30_table29.dat.gz.fits')
#hdulist_eq4 = fits.open(fname_eq4)
#data = hdulist_eq4[1].data
#hdulist_eq4.close() 
#data = data[(data['Teff'] > 12000) & (data['Teff'] < 33000) & (data['logg'] < 5) & (data['logg'] > 3.5) & (data['xi'] <=4)]
#
#hdulist_y = fits.open(fname_y)
#data_y = hdulist_y[1].data
#hdulist_y.close()
#data_y = data_y[(data_y['logTeff'] > np.log10(12000)) & (data_y['logTeff'] < np.log10(33000)) & (data_y['logg'] < 5) & (data_y['logg'] > 3.5) & (data_y['xi'] <=4)]
#lydd = lydClaret(data, data_y)




class predict_parameters():

     def _lydd(self, data_ldc=None, data_gdc=None, model_dic=None):
         from .lyd import lydClaret, xgb_ldyClaret
         if  model_dic is not None:
            lydd = xgb_ldyClaret(model_dic)
         else:
            lydd = lydClaret(data_ldc, data_gdc)
         self.lydd = lydd

     def binarylc(self, time, phase0, per, tc, secosw, sesinw, m1, q, R1, R2, Teff1, Teff2, incl,xi=2, Z = np.log10(0.38), gdc_1=None, gdc_2=None,  **argument):
         if np.any(np.isnan(np.array([per, tc, secosw, sesinw, m1, q, R1, R2, Teff1, Teff2, incl]))): return
         m2 = m1*q
         logg1 = np.log10(m1/R1**2/3.6469715e-5)
         logg2 = np.log10(m2/R2**2/3.6469725e-5)
         L1 = 9.0093545e-16 * R1**2*Teff1**4
         L2 = 9.0093545e-16 * R2**2*Teff2**4
         sma = (m1*(1 + q)* per**2)**(1/3) * 4.2082783
         r_1 = R1/sma
         r_2 = R2/sma
         ld_1= ld_2 = 'claret'
         shape_1 = 'roche_v'
         shape_2 = 'roche_v'
         t_lc = phase0*per+tc
         sbratio = L2/L1
         ldc_1 = self.lydd.interpolate_lyd(np.log10(Teff1), logg1, Z, xi)
         if np.any(np.isnan(ldc_1)):
            print(f'ldc_1 = {ldc_1}')
            return
         ldc_2 = self.lydd.interpolate_lyd(np.log10(Teff2), logg2, Z, xi)
         if np.any(np.isnan(ldc_2)):
            print(f'ldc_2 = {ldc_2}')
            return
         if np.any(np.isnan(ldc_2)): return
         if gdc_2 is None: gdc_2 = ldc_2[4]
         if gdc_1 is None: gdc_1 = ldc_1[4]
         lc_1 = ellc.lc(t_lc,t_zero=tc, q=q, period=per, a = sma,
                        radius_1=r_1, radius_2=r_2,incl=incl,sbratio=sbratio,
                        ld_1=ld_1, ldc_1=list(ldc_1[:4]), gdc_1=gdc_1,
                        ld_2=ld_2, ldc_2=list(ldc_2[:4]), gdc_2=gdc_2,
                        f_c=secosw, f_s=sesinw,
                        shape_1=shape_1,shape_2=shape_2,exact_grav=True, **argument)
         phase = np.mod((time-tc)/per, 1)
         lc = np.interp(phase, phase0, lc_1)
         return lc, L1, L2
     
     def quadruplelc(self, time, phase0,
                     per1, tc1, secosw1, sesinw1, m11, q1, R11, R12, Teff11, Teff12, incl1,
                     per2, tc2, secosw2, sesinw2, m21, q2, R21, R22, Teff21, Teff22, incl2,
                     bfac_1 = 4.96, bfac_2 = 5.01, heat11=0, heat12=0,
                     gdc11=None, gdc12=None, gdc21=None, gdc22=None,
                     tpiv = 0, p1=1, p2=0, p3=0, lc3=0):
         '''
         # Eq. (1) of Southworth 2022 http://arxiv.org/abs/2201.02516
         '''
         nlc = len(time)
         lc1 = np.zeros(nlc)
         lc2 = np.zeros(nlc)
         try:
           lc1, L11, L12 =self.binarylc(time, phase0,per1, tc1, secosw1, sesinw1, m11, q1, R11, R12, Teff11, Teff12, incl1,bfac_1 =bfac_1, bfac_2 = bfac_2,
                                         heat_1=heat11, heat_2=heat12, gdc_1=gdc11, gdc_2=gdc12)
           lc1 = (L11+L12)*lc1
         except:
           print("ellc cann't calculate lc1 by paramerters:", per1, tc1, secosw1, sesinw1, m11, q1, R11, R12, Teff11, Teff12, incl1)
           L11, L12 = 0, 0
         try:
           lc2, L21, L22 =self.binarylc(time, phase0,per2, tc2, secosw2, sesinw2, m21, q2, R21, R22, Teff21, Teff22, incl2, gdc_1=gdc21, gdc_2=gdc22)
           lc2 = (L21+L22)*lc2
         except:
           print("ellc cann't calculate lc2 by paramerters:", per2, tc2, secosw2, sesinw2, m21, q2, R21, R22, Teff21, Teff22, incl2)
           L21, L22 = 0, 0
         lc = (lc1 + lc2+lc3)/(L11+L12+L21+L21+lc3)
         dt = (time-tpiv)
         lc = lc*(p1 + p2*dt + p3*dt**2)
         return lc, L11, L12, L21, L22

     def residual(self, time, phase0, flux, flux_error,
                  per1, tc1, secosw1, sesinw1, m11, q1, R11, R12, Teff11, Teff12, incl1,
                  per2, tc2, secosw2, sesinw2, m21, q2, R21, R22, Teff21, Teff22, incl2, 
                  bfac_1 = 4.96, bfac_2 = 5.01, heat11=0, heat12=0,
                  gdc11=None, gdc12=None, gdc21=None, gdc22=None,
                  tpiv = 0, p1=1, p2=0, p3=0, lc3=0):
         
         lc,  L11, L12, L21, L22 = self.quadruplelc(time, phase0,
                          per1, tc1, secosw1, sesinw1, m11, q1, R11, R12, Teff11, Teff12, incl1,
                          per2, tc2, secosw2, sesinw2, m21, q2, R21, R22, Teff21, Teff22, incl2,
                          bfac_1 =bfac_1, bfac_2 = bfac_2, heat11=heat11, heat12=heat12,
                          gdc11=gdc11, gdc12=gdc12, gdc21=gdc21, gdc22=gdc22,
                          tpiv = tpiv, p1=p1, p2=p2, p3=p3, lc3=lc3)
         lc = lc/np.median(lc) 
         chi = (flux-lc)/flux_error
         return chi


def residualfunc(parameters, time, flux, flux_error, phase0):
    per1, tc1, secosw1, sesinw1, m11, q1, R11, R12, Teff11, Teff12, incl1, per2, tc2, secosw2, sesinw2, m21, q2, R21, R22, Teff21, Teff22, incl2 = paramters
    L11 = 9.0093545e-16 * R11**2*Teff11**4
    L12 = 9.0093545e-16 * R12**2*Teff12**4
    L21 = 9.0093545e-16 * R21**2*Teff21**4
    L22 = 9.0093545e-16 * R22**2*Teff22**4
    alpha = (L21 + L22)/(L11 + L12) 
    deltaratio =  L22/L22
    if alpha > 0.02 or alpha < 0.01: return np.inf
    if deltaratio > 0.34: return np.inf
    chi = pridect.residual(time, phase0, flux, flux_error,
                     per1, tc1, secosw1, sesinw1, m11, q1, R11, R12, Teff11, Teff12, incl1,
                     per2, tc2, secosw2, sesinw2, m21, q2, R21, R22, Teff21, Teff22, incl2,bfac_1 = 4.96, bfac_2 = 5.01)
    return chi

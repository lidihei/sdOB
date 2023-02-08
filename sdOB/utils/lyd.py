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
from scipy import ndimage
import re


'''
example of regli
x = np.arange(2,5)
y = np.arange(1,10)
xx, yy = np.meshgrid(x,y)

X = np.ravel(xx)
Y = np.ravel(yy)
Z = X**2 + Y**2
Z2 = X**3 + Y**2
r = Regli.init_from_flats(np.array([X[:-1], Y[:-1]]).T, verbose=True)
r.set_values(np.array([Z[:-1], Z2[:-1]]).T)
r.interpn(np.array([2.5, 1]))
'''

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

    '''
    examples:
    dire = '/home/lijiao/lijiao/limbdarkening/Claret'
    fname_eq4 = os.path.join(dire, 'J_A+A_600_A30_table27.dat.gz.fits') # downlaod from https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/600/A30#/browse
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
    '''
    def __init__(self, grid_eq4=None, grid_y=None):
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
        if grid_eq4 is not None:
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
        if grid_eq4 is not None:
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

    def interpolate_ldy3(self, teff, logg, Z, fix_xi =2, ldtab=None, ytab=None):
        '''
        interpolate limb-darknening coefficient, doppler beeming factor and gravity-darkening coefficient 
        from Claret et al. 2017
        ##------------------------------------------------
        from sdOB.utils.lyd import claret_tab
        fname_ldc = 'J_A+A_600_A30/table28.dat.gz'  
        # downloaded from https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/600/A30#/browse  
        ldtab = claret_tab()
        _ = ldtab.read_tab2017_ldc(fname_ldc)
        _, _ = ldtab.fix_parameter(index = 3, fix_value=fix_xi)
        _, _ = ldtab.create_pixeltypegrid()
        
        ##--------------------------------------------------
        fname_y = 'J_A+A_600_A30/table29.dat.gz'  
        # downloaded from https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/600/A30#/browse
        ytab = claret_tab()
        _ = ytab.read_tab2017_y(fname_y)
        _, _ = ytab.fix_parameter(index = 3, fix_value=fix_xi)
        _, _ = ytab.create_pixeltypegrid()
        
        paramters:
        ---------------
        teff, logg
        returns:
        a1, a2, a3, a4, y
        ---------------
        
        '''
        if type(teff) !=  np.ndarray: teff = np.array([teff])
        if ldtab is None:
           fname_ldc = 'J_A+A_600_A30/table28.dat.gz'  
           # downloaded from https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/600/A30#/browse
           
           ldtab = claret_tab()
           _ = ldtab.read_tab2017_ldc(fname_ldc)
           _, _ = ldtab.fix_parameter(index = 3, fix_value=fix_xi)
           _, _ = ldtab.create_pixeltypegrid()
        
        if ytab is None:
           fname_y = 'J_A+A_600_A30/table29.dat.gz'  
           # downloaded from https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/600/A30#/browse
           ytab = claret_tab()
           _ = ytab.read_tab2017_y(fname_y)
           _, _ = ytab.fix_parameter(index = 3, fix_value=fix_xi)
           _, _ = ytab.create_pixeltypegrid()
            
        p = np.ones((3, teff.shape[0]))
        p[0] = logg
        p[1] = np.log10(teff)
        p[2] = Z
        a1, a2, a3, a4 = ldtab.interpolate(p, axis_values=None, pixelgrid=None)
        y =ytab.interpolate(p, axis_values=None, pixelgrid=None)
        return a1, a2, a3, a4, y

    def interpolate_ldy_from_DA(self, teff, logg, ldtab=None, ytab=None):
        '''
        interpolate limb-darknening coefficient, doppler beeming factor and gravity-darkening coefficient 
        from Claret et al. 2020 of DA white dwarf models 
        ##------------------------------------------------------------------------- 
        from sdOB.utils.lyd import claret_tab 
        fname = 'Claret_limb_gravity_beaming/2020/Gravity_limb-darkening/TABLE104C' 
        # downloaded frome https://cdsarc.cds.unistra.fr/ftp/J/A+A/634/A93/OriginalTab.tar.gz
        ldtab = claret_tab()
        names, data = ldtab.read_tab2020(fname)
        grid_pars = data[:, [0,1]].T
        indc = [4+5*_ for _ in [0, 1, 2, 3, 6]]
        grid_data = data[:, indc].T
        _, _ = ldtab.create_pixeltypegrid(grid_pars=grid_pars, grid_data=grid_data)
        
        ##------------------------------------------------------------------------------
        fname = 'Claret_limb_gravity_beaming/2020/Gravity_limb-darkening/TABLE105C' 
        # downloaded from https://cdsarc.cds.unistra.fr/ftp/J/A+A/634/A93/OriginalTab.tar.gz
        ytab = claret_tab()
        names, data = ytab.read_tab2020(fname, skiprows=2, skipcolumns=3)
        grid_pars = data[:, [0,1]].T
        indc = [4+5*_ for _ in [0, 1]]
        grid_data = data[:, indc].T
        _, _ = ytab.create_pixeltypegrid(grid_pars=grid_pars, grid_data=grid_data)

        paramters:
        ---------------
        teff, logg
        returns:
        a1, a2, a3, a4, y, bfac
        ---------------
        
        '''
        if type(teff) !=  np.ndarray: teff = np.array([teff])
        if ldtab is None:
           fname = '/home/lijiao/lijiao/Documents/Claret_limb_gravity_beaming/2020/Gravity_limb-darkening/TABLE104C' 
           # downloaded frome https://cdsarc.cds.unistra.fr/ftp/J/A+A/634/A93/OriginalTab.tar.gz
           
           ldtab = claret_tab()
           names, data = ldtab.read_tab2020(fname)
           grid_pars = data[:, [0,1]].T
           indc = [4+5*_ for _ in [0, 1, 2, 3, 6]]
           grid_data = data[:, indc].T
           _, _ = ldtab.create_pixeltypegrid(grid_pars=grid_pars, grid_data=grid_data)
        if ytab is None:
           fname = '/home/lijiao/lijiao/Documents/Claret_limb_gravity_beaming/2020/Gravity_limb-darkening/TABLE105C' 
           # downloaded from https://cdsarc.cds.unistra.fr/ftp/J/A+A/634/A93/OriginalTab.tar.gz
           ytab = claret_tab()
           names, data = ytab.read_tab2020(fname, skiprows=2, skipcolumns=3)
           grid_pars = data[:, [0,1]].T
           indc = [4+5*_ for _ in [0, 1]]
           grid_data = data[:, indc].T
           _, _ = ytab.create_pixeltypegrid(grid_pars=grid_pars, grid_data=grid_data)
            
        p = np.ones((2, teff.shape[0]))
        p[0] = logg
        p[1] = teff
        a1, a2, a3, a4, bfac = ldtab.interpolate(p, axis_values=None, pixelgrid=None)
        y =ytab.interpolate(p, axis_values=None, pixelgrid=None)
        return a1, a2, a3, a4, y, bfac




class claret_tab():

    def __init__(self):
       '''
       read and interpolate tables of limb and gravite-darkening coefficient of Claret
       examples:
       # limb darkening coefficient of TESS from LDCs DA models
       from sdOB.utils.lyd import claret_tab
       fname = 'Gravity_limb-darkening/TABLE104C' # downloaded frome https://cdsarc.cds.unistra.fr/ftp/J/A+A/634/A93/OriginalTab.tar.gz
       ctab = claret_tab()
       names, data = ctab.read_tab2020(fname)
       grid_pars = data[:, [0,1]].T
       indc = [4+5*_ for _ in [0, 1, 2, 3, 6]]
       grid_data = data[:, indc].T
       axis_values, pixelgrid = ctab.create_pixeltypegrid(grid_pars=grid_pars, grid_data=grid_data)
       p = np.array([[5.25], [36000]])
       cc= ctab.interpolate(p, axis_values=None, pixelgrid=None)
       a1, a2, a3, a4, bfac = cc.T[0]
       
       # Gravity darkening coeffiecient of TESS from LDCs DA models
       fname = 'Gravity_limb-darkening/TABLE105C' # downloaded from https://cdsarc.cds.unistra.fr/ftp/J/A+A/634/A93/OriginalTab.tar.gz
       ctab = claret_tab()
       names, data = ctab.read_tab2020(fname, skiprows=2, skipcolumns=3)
       grid_pars = data[:, [0,1]].T
       indc = [4+5*_ for _ in [0, 1]]
       grid_data = data[:, indc].T
       axis_values, pixelgrid = ctab.create_pixeltypegrid(grid_pars=grid_pars, grid_data=grid_data)
       p = np.array([[5.25], [36000]])
       cc= ctab.interpolate(p, axis_values=None, pixelgrid=None)
       y = cc.T[0]
       
       # Limb darkening coeffiecient of TESS from LDCs ATLAS models xi =2
       fname_ldc = 'J_A+A_600_A30/table28.dat.gz'  # downloaded from https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/600/A30#/browse
       ctab = claret_tab()
       _ = ctab.read_tab2017_ldc(fname_ldc)
       _, _ = ctab.fix_parameter(index = 3, fix_value=2)
       axis_values, pixelgrid = ctab.create_pixeltypegrid()
       p = np.array([[5], [np.log10(34999)], [0]])
       cc= ctab.interpolate(p, axis_values=axis_values, pixelgrid=pixelgrid)
       cc.T[0]
       
       
       # Gravity darkening coeffiecient of TESS from LDCs ATLAS models
       fname_y = 'J_A+A_600_A30/table29.dat.gz'  # downloaded from https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/600/A30#/browse
       ctab = claret_tab()
       _ = ctab.read_tab2017_y(fname_y)
       _, _ = ctab.fix_parameter(index = 3, fix_value=2)
       axis_values, pixelgrid = ctab.create_pixeltypegrid()
       p = np.array([[5], [np.log10(34090)], [0]])
       cc= ctab.interpolate(p, axis_values=axis_values, pixelgrid=pixelgrid)
       cc.T[0]
       
       '''

    def read_tab2020(self, fname, skiprows=7, skipcolumns=3):
        '''read tables of limb and gravite-darkening coefficient of Claret et al. 2020, download from 
        https://cdsarc.cds.unistra.fr/ftp/J/A+A/634/A93/OriginalTab.tar.gz
        self.grid_pars: np.array([[logg], [logTeff], [Z]])
        returns:
        ---------------
        names: [list]
        data: [2D array]
        '''
        f = open(fname, 'r')
        line = f.readline()
        names = list(filter(None, re.split(r'\s+', line)))
        
        for _ in np.arange(1, skiprows):
            line = f.readline()
            names += list(filter(None, re.split(r'\s+', line)))[skipcolumns:]
        f.close()
        data = np.loadtxt(fname, skiprows=skiprows)
        shape = data.shape
        data = data.ravel()
        reshape = (int(shape[0]/skiprows), int(shape[1]* skiprows))
        data = data.reshape((reshape))
        indc = list(np.arange(skipcolumns))
        for _i in np.arange(skiprows):
           indc += list(np.arange(skipcolumns, 8)+_i*8)
        data = data[:, indc]
        indpars = list(np.arange(skipcolumns))
        self.grid_pars = data[:, indpars].T
        self.grid_data = data[:, skipcolumns:].T
        return names, data
    
    def read_tab2017_ldc(self, fname_ldc, usecols=(0, 1, 2,3,4,5,6,7)):
        '''
        read tables of limb and gravite-darkening coefficient of Claret 2017,
        Limb and gravity-darkening coefficients for the TESS satellite at several metallicities, 
        surface gravities, and microturbulent velocities
        download from 
        https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/tar.gz?J/A+A/600/A30
        self.grid_pars: np.array([[logg], [logTeff], [Z],[xi]])
        returns:
        -------------
        data: [2D array]
        '''
        data = np.loadtxt(fname_ldc, usecols=usecols)
        shape = data.shape
        #grid_pars = np.empty((shape[0], 4))
        #grid_data = np.empty((shape[0], 4))
        grid_pars = data[:, [0,1,2,3]]#np.empty((shape[0], 4))
        grid_data = data[:, [4,5,6,7]]#np.empty((shape[0], 4))
        grid_pars[:, 1] = np.log10(grid_pars[:, 1])
        self.grid_pars = grid_pars.T
        self.grid_data = grid_data.T
        return data
    
    def fix_parameter(self, grid_pars=None, grid_data=None, index = 3, fix_value=2):
        '''fix the parameter
        paramters:
        ---------------
        grid_pars: [2D array]  grid_pars.shape = (x_n, N), x_n is the number of parameters x
        grid_data: [2D array]  grid_pars.shape = (y_n, N), y_n is the number of y = f(x)
        returns:
        ---------------
        grid_pars1: [2D array]  grid_pars.shape = (x_n-1, Nf); Nf = np.sum(grid_pars[index] == fix_value)
        grid_pars1: [2D array]  grid_pars.shape = (y_n, Nf)
        '''
        if grid_pars is None: grid_pars = self.grid_pars
        if grid_data is None: grid_data = self.grid_data
        _ind = grid_pars[index] == fix_value
        x_n , N = grid_pars.shape
        ind = []
        for _ in np.arange(x_n):
            if _ == index:continue
            ind.append(_)
        grid_data1 = grid_data.T[_ind].T
        grid_pars1 = grid_pars.T[_ind].T[ind]
        self.grid_pars = grid_pars1
        self.grid_data = grid_data1
        return grid_pars1, grid_data1
        
    
    def read_tab2017_y(self, fname_y, usecols=(0, 1, 2,3,4)):
        '''
        read tables of limb and gravite-darkening coefficient of Claret 2017,
        Limb and gravity-darkening coefficients for the TESS satellite at several metallicities, 
        surface gravities, and microturbulent velocities
        download from 
        https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/tar.gz?J/A+A/600/A30
        self.grid_pars: np.array([[logg], [logTeff], [Z],[xi]])
        returns:
        data: [2D array]
        '''
        data = np.loadtxt(fname_y, usecols = usecols)
        shape = data.shape
        #grid_pars = np.empty((shape[0], 4))
        #grid_data = np.empty((shape[0], 4))
        grid_pars = data[:, [2, 3, 0,1]]
        grid_data = data[:, [4]]
        self.grid_pars = grid_pars.T
        self.grid_data = grid_data.T
        return data
    
    def create_pixeltypegrid(self, grid_pars=None, grid_data=None):
        """
        Creates pixelgrid and arrays of axis values.
        
        Starting from:
           * grid_pars: 2D numpy array, 1 column per parameter, unlimited number of cols
           * grid_data: 2D numpy array, 1 column per variable, data corresponding to the rows in grid_pars
        
        example: interpolation in a 3D grid containing stellar evolution models. Say we have as
        input parameters mass, age and metalicity, and want to obtain teff and logg as variables.
        
        grid_pars =
           +------+-----+------+
           | mass | age | Fe/H |
           +------+-----+------+
           | 1.0  | 1.0 | -0.5 |
           +------+-----+------+
           | 2.0  | 1.0 | -0.5 |
           +------+-----+------+
           | 1.0  | 2.0 | -0.5 |
           +------+-----+------+
           | 2.0  | 2.0 | -0.5 |
           +------+-----+------+
           | 1.0  | 1.0 |  0.0 |
           +------+-----+------+
           | 2.0  | 1.0 |  0.0 |
           +------+-----+------+
           |...   |...  |...   |
           +------+-----+------+
           
        grid_data = 
           +------+------+
           | teff | logg |
           +------+------+
           | 5000 | 4.45 |
           +------+------+
           | 6000 | 4.48 |
           +------+------+
           |...   |...   |
           +------+------+
     
        >>> grid_pars = np.array([[ 1. ,  1. , 0.5], [ 2. ,  1. ,  -0.5]]).T  # grid_pars.shape = (x, N)
        >>> grid_data = np.array([[ 5000. , 4.5], [ 6000 , 4.5]]).T   # grid_data.shape  = (y, N)
        >>> axis_values, pixelgrid =  create_pixeltypegrid(grid_pars,grid_data)
                                                                                                                                                                               
        The resulting grid will be rectangular and complete. This means that every
        combination of unique values in grid_pars should exist. If this is not the
        case, a +inf value will be inserted in grid_data at all locations that are 
        missing!
     
        
        :param grid_pars: Npar x Ngrid array of parameters
        :type grid_pars: array
        :param grid_data: Ndata x Ngrid array of data
        :type grid_data: array
        
        :return: axis values and pixelgrid
        :rtype: array, array
       """
        
        if grid_pars is None: grid_pars = self.grid_pars
        if grid_data is None: grid_data = self.grid_data
        uniques = [np.unique(column, return_inverse=True) for column in grid_pars]
        #[0] are the unique values, [1] the indices for these to recreate the original array

        axis_values = [uniques_[0] for uniques_ in uniques]
        unique_val_indices = [uniques_[1] for uniques_ in uniques]
        
        data_dim, data_size = np.shape(grid_data)
     
        par_dims   = [len(uv[0]) for uv in uniques]
     
        par_dims.append(data_dim)
        pixelgrid = np.ones(par_dims)
        
        # We put np.inf as default value. If we get an inf, that means we tried to access
        # a region of the pixelgrid that is not populated by the data table
        pixelgrid[pixelgrid==1] = np.inf
        
        # now populate the multiDgrid
        #indices = [uv[1] for uv in uniques]
        #pixelgrid[indices] = grid_data.T
        indices = np.array([uv[1] for uv in uniques])
        for _i in np.arange(data_size):
            pixelgrid[tuple(indices[:,_i])] = grid_data[:, _i]
        self.axis_values = axis_values
        self.pixelgrid = pixelgrid
        return axis_values, pixelgrid
    
    def interpolate(self, p, axis_values=None, pixelgrid=None):
        """
        Interpolates in a grid prepared by create_pixeltypegrid().
        
        p is an array of parameter arrays
        each collumn contains the value for the corresponding parameter in grid_pars
        each row contains a set of model parameters for wich the interpolated values
        in grid_data are requested.
        
        example: continue with stellar evolution models used in create_pixeltypegrid
        
        p = 
           +------+-----+-------+
           | mass | age | Fe/H  | 
           +------+-----+-------+
           | 1.21 | 1.3 | 0.24  |
           +------+-----+-------+
           | 1.57 | 2.4 | -0.15 |
           +------+-----+-------+
           |...   |...  |...    |
           +------+-----+-------+
          
        >>>  
        >>> p = np.array([[1.21, 1.3, 0.24], [1.57, 2.4, -0.15]])
        >>> interpolate(p, axis_values, pixelgrid)
        >>> some output
        
        :param p: Npar x Ninterpolate array containing the points which to
                  interpolate in axis_values
        :type p: array
        :param axis_values: output from create_pixeltypegrid
        :type axis_values: array
        :param pixelgrid: output from create_pixeltypegrid
        :type pixelgrid: array
        
        :return: Ndata x Ninterpolate array containing the interpolated values
                 in pixelgrid
        :rtype: array
        
        """
        # convert requested parameter combination into a coordinate
        #p_ = [np.searchsorted(av_,val) for av_, val in zip(axis_values,p)]
        # we force the values to be inside the grid, to avoid edge-effect rounding
        # (e.g. 3.099999 is edge, while actually it is 3.1). For values below the
        # lowest value, this is automatically done via searchsorted (it return 0)
        # for values higher up, we need to force it
        
        #p_ = []
        #for av_,val in zip(axis_values,p):
        #   indices = np.searchsorted(av_,val)
        #   indices[indices==len(av_)] = len(av_)-1
        #   p_.append(indices)
        #
     
        #-- The type of p is changes to the same type as in axis_values to catch possible rounding errors
        #   when comparing float64 to float32.
        if axis_values is None: axis_values = self.axis_values
        if pixelgrid is None: pixelgrid = self.pixelgrid
        for i, ax in enumerate(axis_values):
           p[i] = np.array(p[i], dtype = ax.dtype)
        
        #-- Convert requested parameter combination into a coordinate
        p_ = np.array([np.searchsorted(av_,val) for av_, val in zip(axis_values,p)])
        lowervals_stepsize = np.array([[av_[p__-1], av_[p__]-av_[p__-1]] \
                                for av_, p__ in zip(axis_values,p_)])
        p_coord = (p-lowervals_stepsize[:,0])/lowervals_stepsize[:,1] + np.array(p_)-1
        self.p_coord = p_coord
        
        # interpolate
        y = np.array([ndimage.map_coordinates(pixelgrid[...,i],p_coord, order=1, prefilter=False) \
                    for i in range(np.shape(pixelgrid)[-1])])
        return y




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


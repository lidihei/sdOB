from astropy.table import Table
import scipy
import radvel, ellc
import lightkurve as lk
from astropy import constants, units
from radvel.orbit import timeperi_to_timetrans, timetrans_to_timeperi
from ..utils.lc_tools import *
from ..utils.lyd import *
import joblib
import matplotlib.pyplot as plt
from ..utils.binary_orbit_para import eR_Rochelobe
from ..utils.lyd import lydClaret, xgb_ldyClaret
from OANDBH.utils import parsec

def rvfold(t, orbit, vgamma, t0=None):
    ''' calculate rv by orbital parameters and fold it
    parameters:
    -------------
    t: [array]
    orbit: [list] e.g. orbit = [per, tp, ecc, omega, K]  omege in radian
    vgamma: [float], binary systematic velocity
    returns:
    ------------
    rvdens_fold 
    '''
    rvdens = radvel.kepler.rv_drive(t, orbit) + vgamma
    rvdens_lc = lk.LightCurve(time=t, flux=rvdens)
    if t0 is None: t0 = orbit[1]
    rvdens_fold = rvdens_lc.fold(orbit[0], t0)
    return rvdens_lc, rvdens_fold

def getrvresidual(rv_lc, orbit, t0, vgamma):
    '''
    parameters:
    -----------------
    rv_lc: e.g. rv_lc = lk.LightCurve(time=time, flux=flux, flux_err = flux_err); import lightkurve as lk
    orbit: [list] e.g. orbit = [per, tp, ecc, omega, K]  omege in radian
    t0: [float], e.g. tc
    vgamma: [float], binary systematic velocity
    returns:
    -----------------
    rv_resi
    rv_resi_flod
    '''
    rv_resi = rv_lc.flux - radvel.kepler.rv_drive(rv_lc.time, orbit) - vgamma
    rv_resi = lk.LightCurve(time=rv_lc.time, flux=rv_resi, flux_err=rv_lc.flux_err )
    per = orbit[0]
    rv_resi_flod = rv_resi.fold(per, t0)
    return rv_resi, rv_resi_flod


class checkBB():

    def lyd(self, ldcgrid=None, gdcgrid=None, xgbootmodeldic=None):
        if xgbootmodeldic is None:
           lydd = lydClaret(ldcgrid, gdcgrid)
           self.interp_lyd = lydd.interpolate_lyd2
        else:
           lydd = xgb_ldyClaret(xgbootmodeldic)
           self.interp_lyd = lydd.interpolate_lyd

    def parsecMTgR(self, parsecgrid=None, ind=None, xgbootmodeldic=None):
        if xgbootmodeldic is None:
           parseclmh = parsec.parsecLMH(parsecgrid, ind)
           self.MTgR = parseclmh.MTgR
        else:
           parseclmh = parsec.xgb_parsecLMH(xgbootmodeldic)
           self.MTgR = parseclmh.MTgR

    def parsec(self, parsecgrid=None, ind=None, xgbootmodeldic=None):
        if xgbootmodeldic is None:
           parseclmh = parsec.parsecLMH(parsecgrid, ind)
           self.M = parseclmh.M
        else:
           parseclmh = parsec.xgb_parsecLMH(xgbootmodeldic)
           self.M = parseclmh.M

    def pymcsresualt(self, trace, pers=None, tcs=None, L1s=None, sbratios=None, incls = None, MHs=np.log10(0.38),
                     sesinws=0, secosws=0, R1s=None, R2s=None):
        self.trace = trace
        self.pers = trace.posterior.per.values.ravel() if (pers is None) else pers
        self.tcs =  trace.posterior.tc.values.ravel() if (tcs is None) else tcs
        self.L1s = trace.posterior.L1.values.ravel() if (L1s is None) else L1s
        self.R1s = trace.posterior.R1.values.ravel() if (R1s is None) else R1s
        self.R2s = trace.posterior.R2.values.ravel() if (R2s is None) else R2s
        self.sbratios = trace.posterior.sbratio.values.ravel() if (sbratios is None) else sbratios
        self.incls = trace.posterior.incl.values.ravel() if (incls is None) else incls
        self.L2s = self.L1s * self.sbratios
        self.bestind = np.argmax(self.trace.log_likelihood.likelihood_lc.values.ravel())
        m1s = self.M(self.L1s, MHs)
        m2s = self.M(self.L2s, MHs)
        RR1s = self.R1s**2
        RR2s = self.R2s**2
        logg1s = np.log10(m1s/3.6469715e-5/RR1s)
        logg2s = np.log10(m2s/3.6469715e-5/RR2s)
        Teff1s = (self.L1s/9.0093545e-16/ RR1s)**(0.25)
        Teff2s = (self.L2s/9.0093545e-16/ RR2s)**(0.25)
        self.m1s = m1s
        self.Teff1s = Teff1s
        self.logg1s = logg1s
        self.m2s = m2s
        self.Teff2s = Teff2s
        self.logg2s = logg2s
        self.qs = m2s/m1s
        self.smas =  (self.m1s*(1 + self.qs)* self.pers**2)**(1/3) * 4.2082783
        self.sesinws = sesinws
        self.secosws = secosws
        self.eccs = self.secosws**2 + self.sesinws**2
        self.omega_rads =  np.arctan2(self.sesinws, self.secosws)
        self.sma1s = self.smas/(1+1/self.qs)
        self.R_L1s =eR_Rochelobe(1/self.qs) * self.smas
        self.R_L2s =eR_Rochelobe(self.qs) * self.smas
        #self.r_1s = self.R1s/self.smas
        #self.r_2s = self.R2s/self.smas
        self.tps = timetrans_to_timeperi(self.tcs, self.pers, self.eccs, self.omega_rads)


    def print_best_values(self):
        #---------------------------best point------------------------------------------------------
        argmax = self.bestind
        _per, _tc, _tp = self.pers[argmax], self.tcs[argmax], self.tps[argmax]
        _secosw, _sesinw, _ecc, _omega_rad = self.secosws[argmax], self.sesinws[argmax], self.eccs[argmax], self.omega_rads[argmax]
        _vgamma, _jit, _k1, _k2,= self.vgammas[argmax], self.jits[argmax],self.k1s[argmax], self.k2s[argmax]
        _R1, _R2tR1, _R2, _sma =  self.R1s[argmax], self.R2tR1s[argmax], self.R2s[argmax], self.smas[argmax]
        _m1, _m2, _q = self.m1s[argmax], self.m2s[argmax], self.qs[argmax]
        _L1, _L2 = self.L1s[argmax], self.L2s[argmax]
        _incl, _r_1, _r_2,   = self.incls[argmax], self.r_1s[argmax], self.r_2s[argmax]
        _vsini1 = self.vsini1s[argmax]
        _sbratio= self.sbratios[argmax]
        print('----------------------best values----------------------------------')
        print(f'm1, m2, q = {_m1}, {_m2}, {_q} ')
        print(f'per, tc, tp  = {_per}, {_tc}, {_tp}')
        print(f'ecc, omega, secosw, sesinw, incl= {_ecc}, {_omega_rad}, {_secosw}, {_sesinw},  {_incl}')
        print(f'k1, k2, vgamma, jit, vsini1 = {_k1}, {_k2}, {_vgamma}, {_jit}, {_vsini1}')
        print(f'sma, R1, R2, R2tR1, r_1, r_2 = {_sma}, {_R1}, {_R2}, {_R2tR1}, {_r_1}, {_r_2}')
        print(f'L1, L2, sbratio = {_L1}, {_L2}, {_sbratio}')
        print('-----------------------------------------------------------------')
            
    
    def plot_corner(self, samples =None,  cornerstri= None, show=True):
        import corner
        if samples is None:
           self.samples = np.vstack((self.L1s, self.sbratios, self.incls)).T 
        else:
            self.samples = samples
        if cornerstri is None:  cornerstri = [r'$L1$', r'$sbratio$', r'$i$']
        if show:
           fig = corner.corner(
                self.samples,
                labels=cornerstri,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                #**hist2dkwargs,)
           )
        
    def ellc_lc(self, lc, tc=None, q=None, per=None, sma=None,
              secosw=None, sesinw=None, incl=None,
              r_1=None, r_2=None, sbratio=None,
              Teff1 =None, Teff2 = None, logg1= None, logg2=None,
              tdens = None, LBO1=63410.105, LBO2=4576.37, L1=None, L2=None,
              bfac_1 = 0, bfac_2=0,
              bins_len= 0.01, bins=None, shape_1 = 'sphere',  shape_2 = 'sphere',
              ld_1='claret', ld_2 = 'claret', bestind =True,
              ldc_1 = None, gdc_1 = None,
              ldc_2 = None, gdc_2 = None, heat_1=0, heat_2=0,
               show=True, verbose=False,
              foldmethod='mean'):
        '''lc is object of lightkurve, e.g. lc = lk.LightCurve(time=time, flux=flux)
        returns:
        --------
        lc_dens
        lc_resi_bins
        lc_primary_foldsmooth
        lc_primary_fold
        '''
        if tc is None: tc = self.tcs[self.bestind] if bestind else np.median(self.tcs)
        if per is None: per = self.pers[self.bestind] if bestind else np.median(self.pers)
        if q is None: q = self.qs[self.bestind] if bestind else np.median(self.qs)
        if secosw is None: secosw = self.secosws[self.bestind] if bestind else np.median(self.secosws)
        if sesinw is None: sesinw = self.sesinws[self.bestind] if bestind else np.median(self.sesinws)
        if incl is None: incl = self.incls[self.bestind] if bestind else np.median(self.incls)
        if r_1 is None: r_1 = self.r_1s[self.bestind] if bestind else np.median(self.r_1s)
        if r_2 is None: r_2 = self.r_2s[self.bestind] if bestind else np.median(self.r_2s)
        if sma is None: sma = self.smas[self.bestind] if bestind else np.median(self.smas)
        if sbratio is None: sbratio = self.sbratios[self.bestind] if bestind else np.median(self.sbratios)
        if Teff1 is None: Teff1 = self.Teff1s[self.bestind] if bestind else np.median(self.Teff1s)
        if Teff2 is None: Teff2 = self.Teff2s[self.bestind] if bestind else np.median(self.Teff2s)
        if logg1 is None: logg1 = self.logg1s[self.bestind] if bestind else np.median(self.logg1s)
        if logg2 is None: logg2 = self.logg2s[self.bestind] if bestind else np.median(self.logg2s)
        if L1 is None: L1 = self.L1s[self.bestind] if bestind else np.median(self.L1s)
        if L2 is None: L2 = self.L2s[self.bestind] if bestind else np.median(self.L2s)
        if tdens is None: tdens = np.arange(0,4, 0.005)
        ldc1 =self.interp_lyd(np.log10(Teff1), logg1, np.log10(0.38))
        ldc2 =self.interp_lyd(np.log10(Teff2), logg2, np.log10(0.38))
        if bins is None: bins=np.arange(-0.5-bins_len/2, 0.5+bins_len/2, bins_len)
        lc_fold, lc_foldsmooth, fluxmedian  = bin_foldlc2(lc, bins, per, tc, method=foldmethod)
        self.fluxmedian = fluxmedian
        t_lc = lc_foldsmooth.time*per+tc
        fluxbins = lc_foldsmooth.flux
        fluxbinserr = lc_foldsmooth.flux_err
        if verbose:
           print(f'q = {q}; per = {per}; sma={sma}; r_1 = {r_1}; r_2={r_2}; secosw={secosw}; sesinw={sesinw}')
        if ldc_1 is None: ldc_1=list(ldc1[:4])
        if ldc_2 is None: ldc_2=list(ldc2[:4])
        if gdc_1 is None: gdc_1=ldc1[4]
        if gdc_2 is None: gdc_2=ldc2[4]
        lc_0 = ellc.lc(lc.time,t_zero=tc, q=q, period=per,
                   a = sma, bfac_1 = bfac_1,bfac_2 = bfac_2,
                   radius_1=r_1, radius_2=r_2,incl=incl,sbratio=sbratio,
                   ld_1=ld_1, ldc_1=ldc_1, gdc_1=gdc_1,
                   ld_2=ld_2, ldc_2=ldc_2, gdc_2=gdc_2,heat_1=heat_1, heat_2=heat_2,
                   f_c=secosw, f_s=sesinw,
                   shape_1=shape_1,shape_2=shape_2,exact_grav=True)
        lc_1 = ellc.lc(t_lc,t_zero=tc, q=q, period=per,
                   a = sma, bfac_1 = bfac_1,bfac_2 = bfac_2,
                   radius_1=r_1, radius_2=r_2,incl=incl,sbratio=sbratio,
                   ld_1=ld_1, ldc_1=ldc_1, gdc_1=gdc_1,
                   ld_2=ld_2, ldc_2=ldc_2, gdc_2=gdc_2,
                   f_c=secosw, f_s=sesinw,
                   shape_1=shape_1,shape_2=shape_2,exact_grav=True, heat_1=heat_1, heat_2=heat_2)
        lcdens = ellc.lc(tdens,t_zero=tc, q=q, period=per,
                   a = sma, bfac_1 = bfac_1,bfac_2 = bfac_2,
                   radius_1=r_1, radius_2=r_2,incl=incl,sbratio=sbratio,
                   ld_1=ld_1, ldc_1=ldc_1, gdc_1=gdc_1,
                   ld_2=ld_2, ldc_2=ldc_2, gdc_2=gdc_2,
                   f_c=secosw, f_s=sesinw,
                   shape_1=shape_1,shape_2=shape_2,exact_grav=True,heat_1=heat_1, heat_2=heat_2 )
        self.lctime = lc_0
        self.time = lc.time
        self.lctdens = lcdens
        self.tdens = tdens
        LBO = LBO1+ LBO2
        LBB = L1+L2
        lc_0 = (lc_0*LBB + LBO)/(LBO)
        lc_1 = (lc_1*LBB + LBO)/(LBO)
        lcdens = (lcdens*LBB + LBO)/(LBO)
        medianlc = np.median(lc_1)
        lcdens = lcdens/medianlc
        lc_1 = lc_1/medianlc
        lcdens = lk.LightCurve(time=tdens, flux=lcdens)
        lc_resi_bins = lk.LightCurve(time=t_lc, flux=fluxbins - lc_1, flux_err= fluxbinserr)
        lc_resi =lc.copy()
        lc_resi.flux = lc_resi.flux/np.median(lc_resi.flux)
        lc_resi.flux = lc_resi.flux - lc_0/np.median(lc_0)
        lcdens_fold = lcdens.fold(per, tc)
        lc_resi_bins_fold = lc_resi_bins.fold(per, tc)
        lc_resi_fold = lc_resi.fold(per, tc)
        self.lcdens = lcdens; self.lcdens_fold = lcdens_fold
        self.lc_resi_bins = lc_resi_bins; self.lc_resi_bins_fold= lc_resi_bins_fold
        self.lc_resi = lc_resi; self.lc_resi_fold = lc_resi_fold
        self.ellcfluxmedian = medianlc
        self.lc_0 = lc_0
        self.lc_fold =lc_fold 
        if show:
           fig, axs = plt.subplots(2,1, figsize=(8.485, 6.75), sharex=True, gridspec_kw={'height_ratios': [2,1]})
           plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0)
           ax = axs[0]
           lcdens_fold.plot(ax=ax, c='r', lw=2)
           print(len(lcdens_fold.time))
           #lc_foldsmooth.errorbar(ax=ax, c='k',fmt='o', ms=5)
           lc_fold.scatter(ax=ax)
           plt.sca(axs[1])
           ax = axs[1]
           #lc_resi_bins_fold.errorbar(ax=ax, fmt='o', ms=5, c='k')
           lc_resi_fold.scatter(ax=ax)
           plt.axhline(y=0)
        return lcdens, lc_resi_bins, lc_foldsmooth,lc_fold

    def ellc_lc1(self, lc, tc=None, q=None, per=None, sma=None,
              secosw=None, sesinw=None, incl=None, R1 = None, R2 = None,
              r_1=None, r_2=None, sbratio=None,
              Teff1 =None, Teff2 = None, logg1= None, logg2=None,
              tdens = None, LBO1=63410.105, LBO2=4576.37, L1=None, L2=None,
              bfac_1 = 0, bfac_2=0, MH = np.log10(0.38),
              bins_len= 0.01, bins=None, shape_1 = 'sphere',  shape_2 = 'sphere',
              ld_1='claret', ld_2 = 'claret', bestind =True,
              ldc_1 = None, gdc_1 = None,
              ldc_2 = None, gdc_2 = None, heat_1=0, heat_2=0,
               show=True, verbose=False,
              foldmethod='mean'):
        '''lc is object of lightkurve, e.g. lc = lk.LightCurve(time=time, flux=flux)
        returns:
        --------
        lc_dens
        lc_resi_bins
        lc_primary_foldsmooth
        lc_primary_fold
        '''
        if tc is None: tc = self.tcs[self.bestind] if bestind else np.median(self.tcs)
        if per is None: per = self.pers[self.bestind] if bestind else np.median(self.pers)
        if secosw is None: secosw = self.secosws[self.bestind] if bestind else np.median(self.secosws)
        if sesinw is None: sesinw = self.sesinws[self.bestind] if bestind else np.median(self.sesinws)
        if incl is None: incl = self.incls[self.bestind] if bestind else np.median(self.incls)
        if L1 is None: L1 = self.L1s[self.bestind] if bestind else np.median(self.L1s)
        if R1 is None: R1 = self.R1s[self.bestind] if bestind else np.median(self.R1s)
        if R2 is None: R2 = self.R2s[self.bestind] if bestind else np.median(self.R2s)
        if sbratio is None: sbratio = self.sbratios[self.bestind] if bestind else np.median(self.sbratios)
        L2 = L1*sbratio
        m1 = self.M(L1, MH)
        m2 = self.M(L2, MH)
        RR1 = R1**2; RR2 = R2**2
        logg1 = np.log10(m1/3.6469715e-5/RR1)
        logg2 = np.log10(m2/3.6469715e-5/RR2)
        Teff1 = (L1/9.0093545e-16/ RR1)**(0.25)
        Teff2 = (L2/9.0093545e-16/ RR2)**(0.25)
        self.m1 = m1
        self.Teff1 = Teff1
        self.logg1 = logg1
        self.R1 = R1
        self.m2 = m2
        self.Teff2 = Teff2
        self.logg2 = logg2
        self.R2 = R2
        self.q = m2/m1
        self.sma =  (self.m1*(1 + self.q)* per**2)**(1/3) * 4.2082783
        self.sesinw = sesinw
        self.secosw = secosw
        self.R_L1 =eR_Rochelobe(1/self.q) * self.sma
        self.R_L2 =eR_Rochelobe(self.q) * self.sma
        self.r_1 = self.R1/self.sma
        self.r_2 = self.R2/self.sma
        r_1 = self.r_1; r_2 = self.r_2
        sma = self.sma; q = self.q
        if tdens is None: tdens = np.arange(0,4, 0.005)
        ldc1 =self.interp_lyd(np.log10(Teff1), logg1, np.log10(0.38))
        ldc2 =self.interp_lyd(np.log10(Teff2), logg2, np.log10(0.38))
        if bins is None: bins=np.arange(-0.5-bins_len/2, 0.5+bins_len/2, bins_len)
        lc_fold, lc_foldsmooth, fluxmedian  = bin_foldlc2(lc, bins, per, tc, method=foldmethod)
        self.fluxmedian = fluxmedian
        t_lc = lc_foldsmooth.time*per+tc
        fluxbins = lc_foldsmooth.flux
        fluxbinserr = lc_foldsmooth.flux_err
        if verbose:
           print(f'q = {q}; per = {per}; sma={sma}; r_1 = {r_1}; r_2={r_2}; secosw={secosw}; sesinw={sesinw}')
        if ldc_1 is None: ldc_1=list(ldc1[:4])
        if ldc_2 is None: ldc_2=list(ldc2[:4])
        if gdc_1 is None: gdc_1=ldc1[4]
        if gdc_2 is None: gdc_2=ldc2[4]
        lc_0 = ellc.lc(lc.time,t_zero=tc, q=q, period=per,
                   a = sma, bfac_1 = bfac_1,bfac_2 = bfac_2,
                   radius_1=r_1, radius_2=r_2,incl=incl,sbratio=sbratio,
                   ld_1=ld_1, ldc_1=ldc_1, gdc_1=gdc_1,
                   ld_2=ld_2, ldc_2=ldc_2, gdc_2=gdc_2,heat_1=heat_1, heat_2=heat_2,
                   f_c=secosw, f_s=sesinw,
                   shape_1=shape_1,shape_2=shape_2,exact_grav=True)
        lc_1 = ellc.lc(t_lc,t_zero=tc, q=q, period=per,
                   a = sma, bfac_1 = bfac_1,bfac_2 = bfac_2,
                   radius_1=r_1, radius_2=r_2,incl=incl,sbratio=sbratio,
                   ld_1=ld_1, ldc_1=ldc_1, gdc_1=gdc_1,
                   ld_2=ld_2, ldc_2=ldc_2, gdc_2=gdc_2,
                   f_c=secosw, f_s=sesinw,
                   shape_1=shape_1,shape_2=shape_2,exact_grav=True, heat_1=heat_1, heat_2=heat_2)
        lcdens = ellc.lc(tdens,t_zero=tc, q=q, period=per,
                   a = sma, bfac_1 = bfac_1,bfac_2 = bfac_2,
                   radius_1=r_1, radius_2=r_2,incl=incl,sbratio=sbratio,
                   ld_1=ld_1, ldc_1=ldc_1, gdc_1=gdc_1,
                   ld_2=ld_2, ldc_2=ldc_2, gdc_2=gdc_2,
                   f_c=secosw, f_s=sesinw,
                   shape_1=shape_1,shape_2=shape_2,exact_grav=True,heat_1=heat_1, heat_2=heat_2 )
        self.lctime = lc_0
        self.time = lc.time
        self.lctdens = lcdens
        self.tdens = tdens
        LBO = LBO1+ LBO2
        LBB = L1+L2
        lc_0 = (lc_0*LBB + LBO)/(LBO)
        lc_1 = (lc_1*LBB + LBO)/(LBO)
        lcdens = (lcdens*LBB + LBO)/(LBO)
        medianlc = np.median(lc_1)
        lcdens = lcdens/medianlc
        lc_1 = lc_1/medianlc
        lcdens = lk.LightCurve(time=tdens, flux=lcdens)
        lc_resi_bins = lk.LightCurve(time=t_lc, flux=fluxbins - lc_1, flux_err= fluxbinserr)
        lc_resi =lc.copy()
        lc_resi.flux = lc_resi.flux/np.median(lc_resi.flux)
        lc_resi.flux = lc_resi.flux - lc_0/np.median(lc_0)
        lcdens_fold = lcdens.fold(per, tc)
        lc_resi_bins_fold = lc_resi_bins.fold(per, tc)
        lc_resi_fold = lc_resi.fold(per, tc)
        self.lcdens = lcdens; self.lcdens_fold = lcdens_fold
        self.lc_resi_bins = lc_resi_bins; self.lc_resi_bins_fold= lc_resi_bins_fold
        self.lc_resi = lc_resi; self.lc_resi_fold = lc_resi_fold
        self.ellcfluxmedian = medianlc
        self.lc_0 = lc_0
        self.lc_fold =lc_fold
        if show:
           fig, axs = plt.subplots(2,1, figsize=(8.485, 6.75), sharex=True, gridspec_kw={'height_ratios': [2,1]})
           plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0)
           ax = axs[0]
           lcdens_fold.plot(ax=ax, c='r', lw=2)
           print(len(lcdens_fold.time))
           #lc_foldsmooth.errorbar(ax=ax, c='k',fmt='o', ms=5)
           lc_fold.scatter(ax=ax)
           plt.sca(axs[1])
           ax = axs[1]
           #lc_resi_bins_fold.errorbar(ax=ax, fmt='o', ms=5, c='k')
           lc_resi_fold.scatter(ax=ax)
           plt.axhline(y=0)
        return lcdens, lc_resi_bins, lc_foldsmooth,lc_fold

    def check_rv(self, t1, rv1, rv1err, t2, rv2, rv2err,
              tp=None, q=None, per=None, omega_rad=None, k1 = None,ecc = None, vgamma=None,
              tdens = None, tc=None,
              bestind = True, show=True):
        '''
        returns:
        --------
        rv1_resi
        rv2_resi
        '''
        if tc is None: tc = self.tps[self.bestind] if bestind else np.median(self.tcs)
        if per is None: per = self.pers[self.bestind] if bestind else np.median(self.pers)
        if q is None: q = self.qs[self.bestind] if bestind else np.median(self.qs)
        if omega_rad is None: omega_rad = self.omega_rads[self.bestind] if bestind else np.median(self.omega_rads)
        if k1 is None: k1 = self.k1s[self.bestind] if bestind else np.median(self.k1s)
        if ecc is None: ecc = self.eccs[self.bestind] if bestind else np.median(self.eccs)
        if vgamma is None: vgamma = self.vgammas[self.bestind] if bestind else np.median(self.vgammas)
        if tdens is None: tdens = np.arange(0,4, 0.005)
        if tp is None: 
           tp = timetrans_to_timeperi(tc, per, ecc, omega_rad)
        else:
           tc = timeperi_to_timetrans(tp, per, ecc, omega_rad)
        orbit1 = [per, tp, ecc, omega_rad, k1 ]
        orbit2 = [per, tp, ecc, omega_rad+np.pi, k1/q ]
        _, rv1dens_fold = rvfold(tdens, orbit1, vgamma, t0=tc)
        _, rv2dens_fold = rvfold(tdens, orbit2, vgamma, t0=tc)
        #rv1_lc = lk.LightCurve(time=mcmc_dic['t1'], flux=mcmc_dic['rv1'], flux_err=mcmc_dic['rv1_err'])
        #rv2_lc = lk.LightCurve(time=mcmc_dic['t2'], flux=mcmc_dic['rv2'], flux_err=mcmc_dic['rv2_err'])
        rv1_lc = lk.LightCurve(time=t1, flux=rv1, flux_err=rv1err)
        rv2_lc = lk.LightCurve(time=t2, flux=rv2, flux_err=rv2err)
        rv1_lc_fold = rv1_lc.fold(per, tc)
        rv2_lc_fold = rv2_lc.fold(per, tc)
        rv1_resi, rv1_resi_flod =  getrvresidual(rv1_lc, orbit1, tc, vgamma)
        rv2_resi, rv2_resi_flod =  getrvresidual(rv2_lc, orbit2, tc, vgamma)
        if show:
           fig, axs = plt.subplots(3,1, figsize=(8.485, 9), sharex=True, gridspec_kw={'height_ratios': [2,1, 1]})
           plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0)
           ax = axs[0]
           rv1dens_fold.plot(ax=ax, c='r')
           rv2dens_fold.plot(ax=ax, c='r')
           rv1_lc_fold.errorbar(ax=ax, c='k',fmt='o', ms=5)
           rv2_lc_fold.errorbar(ax=ax, c='b',fmt='o', ms=5)
           plt.sca(axs[0])
           plt.axhline(y = vgamma, ls='--', color='k')
           plt.ylabel('v (km/s)')
           #plt.ylim([-250, 220])
           plt.sca(axs[1])
           ax = axs[1]
           rv1_resi_flod.errorbar(ax=ax, fmt='o', ms=5, c='k')
           plt.axhline(y=0)
           plt.ylabel(r'$v_{1res}$ (km/s)')
           #plt.ylim(-45, 45)
           plt.sca(axs[2])
           ax = axs[2]
           rv2_resi_flod.errorbar(ax=ax, c='b',fmt='o', ms=5)
           plt.axhline(y=0)
           plt.ylim(-49, 49)
           plt.ylabel(r'$v_{2res}$ (km/s)')
           plt.xlabel('Phase')
        return rv1_resi, rv2_resi, rv1dens_fold, rv2dens_fold



def check_rv(t1, rv1, rv1err, t2, rv2, rv2err,
          tp=None, tc=None, q=None, per=None, omega_rad=None, k1 = None,ecc = None, vgamma=None,
          tdens = None, show=True):
    '''
    returns:
    --------
    rv1_resi
    rv2_resi
    '''
    if tdens is None: tdens = np.arange(0,4, 0.005)
    if tc is None: tc = timeperi_to_timetrans(tp, per, ecc, omega_rad)
    if tp is None: tp = timetrans_to_timeperi(tc, per, ecc, omega_rad) 
    orbit1 = [per, tp, ecc, omega_rad, k1 ]
    orbit2 = [per, tp, ecc, omega_rad+np.pi, k1/q ]
    _, rv1dens_fold = rvfold(tdens, orbit1, vgamma, t0=tc)
    _, rv2dens_fold = rvfold(tdens, orbit2, vgamma, t0=tc)
    #rv1_lc = lk.LightCurve(time=mcmc_dic['t1'], flux=mcmc_dic['rv1'], flux_err=mcmc_dic['rv1_err'])
    #rv2_lc = lk.LightCurve(time=mcmc_dic['t2'], flux=mcmc_dic['rv2'], flux_err=mcmc_dic['rv2_err'])
    rv1_lc = lk.LightCurve(time=t1, flux=rv1, flux_err=rv1err)
    rv2_lc = lk.LightCurve(time=t2, flux=rv2, flux_err=rv2err)
    rv1_lc_fold = rv1_lc.fold(per, tc)
    rv2_lc_fold = rv2_lc.fold(per, tc)
    rv1_resi, rv1_resi_flod =  getrvresidual(rv1_lc, orbit1, tc, vgamma)
    rv2_resi, rv2_resi_flod =  getrvresidual(rv2_lc, orbit2, tc, vgamma)
    if show:
       fig, axs = plt.subplots(3,1, figsize=(8.485, 9), sharex=True, gridspec_kw={'height_ratios': [2,1, 1]})
       plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0)
       ax = axs[0]
       rv1dens_fold.plot(ax=ax, c='r')
       rv2dens_fold.plot(ax=ax, c='r')
       rv1_lc_fold.errorbar(ax=ax, c='k',fmt='o', ms=5)
       rv2_lc_fold.errorbar(ax=ax, c='b',fmt='o', ms=5)
       plt.sca(axs[0])
       plt.axhline(y = vgamma, ls='--', color='k')
       plt.ylabel('v (km/s)')
       #plt.ylim([-250, 220])
       plt.sca(axs[1])
       ax = axs[1]
       rv1_resi_flod.errorbar(ax=ax, fmt='o', ms=5, c='k')
       plt.axhline(y=0)
       plt.ylabel(r'$v_{1res}$ (km/s)')
       #plt.ylim(-45, 45)
       plt.sca(axs[2])
       ax = axs[2]
       rv2_resi_flod.errorbar(ax=ax, c='b',fmt='o', ms=5)
       plt.axhline(y=0)
       plt.ylim(-49, 49)
       plt.ylabel(r'$v_{2res}$ (km/s)')
       plt.xlabel('Phase')
    return rv1_resi, rv2_resi, rv1dens_fold, rv2dens_fold

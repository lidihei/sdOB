import numpy as np
from astropy import units
import astropy.coordinates as coord
SkyCoord = coord.SkyCoord
from astropy.coordinates import FK5


def gal_uvw(dis = None, lsr = False, ra=None, dec=None,
             pmra = None, pmra_err = None, 
             pmdec=None, pmdec_err = None, 
             vrad = None, vrad_err = None, 
             plx  = None, plx_err = None):
    # NAME:
    #     gal_uvw
    # PURPOSE:
    #     Calculate the Galactic space velocity (U,V,W) of star  
    # EXPLANATION:
    #     Calculates the Galactic space velocity U, V, W of star given its 
    #     (1) coordinates, (2) proper motion, (3) distance (or parallax), and 
    #     (4) radial velocity.
    # CALLING SEQUENCE:
    #     GAL_UVW, U, V, W, [/LSR, RA=, DEC=, PMRA= ,PMDEC=, VRAD= , DISTANCE= 
    #              PLX= ]
    # OUTPUT PARAMETERS:
    #      U - Velocity (km/s) positive toward the Galactic *anti*center
    #      V - Velocity (km/s) positive in the direction of Galactic rotation
    #      W - Velocity (km/s) positive toward the North Galactic Pole 
    # REQUIRED INPUT KEYWORDS:
    #      User must supply a position, proper motion,radial velocity and distance
    #      (or parallax).    Either scalars or vectors can be supplied.
    #     (1) Position:
    #      RA - Right Ascension in *Degrees*
    #      Dec - Declination in *Degrees*
    #     (2) Proper Motion
    #      PMRA = Proper motion in RA in arc units (typically milli-arcseconds/yr)
    #            If given mu_alpha --proper motion in seconds of time/year - then
    #             this is equal to 15*mu_alpha*cos(dec)
    #      PMDEC = Proper motion in Declination (typically mas/yr)
    #     (3) Radial Velocity
    #      VRAD = radial velocity in km/s
    #     (4) Distance or Parallax
    #      DISTANCE - distance in parsecs 
    #                 or
    #      PLX - parallax with same distance units as proper motion measurements
    #            typically milliarcseconds (mas)
    #
    # OPTIONAL INPUT KEYWORD:
    #      /LSR - If this keyword is set, then the output velocities will be
    #             corrected for the solar motion (U,V,W)_Sun = (-8.5, 13.38, 6.49) 
    #            (Coskunoglu et al. 2011 MNRAS) to the local standard of rest.
    #            Note that the value of the solar motion through the LSR remains
    #            poorly determined.
    #  EXAMPLE:
    #      (1) Compute the U,V,W coordinates for the halo star HD 6755.  
    #          Use values from Hipparcos catalog, and correct to the LSR
    #      ra = ten(1,9,42.3)*15.    & dec = ten(61,32,49.5)
    #      pmra = 628.42  &  pmdec = 76.65         ;mas/yr
    #      dis = 139    &  vrad = -321.4
    #      gal_uvw,u,v,w,ra=ra,dec=dec,pmra=pmra,pmdec=pmdec,vrad=vrad,dis=dis,/lsr
    #          ===>  u=141.2  v = -491.7  w = 93.9        ;km/s
    #
    #      (2) Use the Hipparcos Input and Output Catalog IDL databases (see 
    #      http://idlastro.gsfc.nasa.gov/ftp/zdbase/) to obtain space velocities
    #      for all stars within 10 pc with radial velocities > 10 km/s
    #
    #      dbopen,'hipp_new,hic'      ;Need Hipparcos output and input catalogs
    #      list = dbfind('plx>100,vrad>10')      ;Plx > 100 mas, Vrad > 10 km/s
    #      dbext,list,'pmra,pmdec,vrad,ra,dec,plx',pmra,pmdec,vrad,ra,dec,plx
    #      ra = ra*15.                 ;Need right ascension in degrees
    #      GAL_UVW,u,v,w,ra=ra,dec=dec,pmra=pmra,pmdec=pmdec,vrad=vrad,plx = plx 
    #      forprint,u,v,w              ;Display results
    # METHOD:
    #      Follows the general outline of Johnson & Soderblom (1987, AJ, 93,864)
    #      except that U is positive outward toward the Galactic *anti*center, and 
    #      the J2000 transformation matrix to Galactic coordinates is taken from  
    #      the introduction to the Hipparcos catalog.   
    # REVISION HISTORY:
    #      Written, W. Landsman                       December   2000
    #      fix the bug occuring if the input arrays are longer than 32767
    #        and update the Sun velocity           Sergey Koposov June 2008
    #	   vectorization of the loop -- performance on large arrays 
    #        is now 10 times higher                Sergey Koposov December 2008
    #      More recent value of solar motion WL/SK   Jan 2011
    #-
    if np.all(dis) != None:
       plx = 1e3/dis         #Parallax in milli-arcseconds
    else: plx = plx
    if np.all(plx_err) == None:
       plx_err = np.nan
    RADEG = 180./np.pi
    cosd = np.cos(dec/RADEG)
    sind = np.sin(dec/RADEG)
    cosa = np.cos(ra/RADEG)
    sina = np.sin(ra/RADEG)
    
    k = 4.74047     #Equivalent of 1 A.U/yr in km/s   
    A_G1 = np.array([ [ -0.0548755604, +0.4941094279, -0.8676661490], 
                     [ -0.8734370902, -0.4448296300, -0.1980763734],
                     [ -0.4838350155,  0.7469822445, +0.4559837762] ])
    A_G = A_G1.T
    vec1 = vrad
    vec1_err2 = vrad_err**2
    vec2 = k*pmra/plx
    vec2_err2 = k**2*(np.square(pmra_err/plx) +np.square(plx_err*pmra/plx**2))
    vec3 = k*pmdec/plx
    vec3_err2 = k**2*(np.square(pmdec_err/plx) +np.square(plx_err*pmdec/plx**2))
   
    u = ( A_G[0,0]*cosa*cosd+A_G[0,1]*sina*cosd+A_G[0,2]*sind)*vec1+\
        (-A_G[0,0]*sina     +A_G[0,1]*cosa                   )*vec2+\
        (-A_G[0,0]*cosa*sind-A_G[0,1]*sina*sind+A_G[0,2]*cosd)*vec3
    v = ( A_G[1,0]*cosa*cosd+A_G[1,1]*sina*cosd+A_G[1,2]*sind)*vec1+\
        (-A_G[1,0]*sina     +A_G[1,1]*cosa                   )*vec2+\
        (-A_G[1,0]*cosa*sind-A_G[1,1]*sina*sind+A_G[1,2]*cosd)*vec3
    w = ( A_G[2,0]*cosa*cosd+A_G[2,1]*sina*cosd+A_G[2,2]*sind)*vec1+\
        (-A_G[2,0]*sina     +A_G[2,1]*cosa                   )*vec2+\
        (-A_G[2,0]*cosa*sind-A_G[2,1]*sina*sind+A_G[2,2]*cosd)*vec3
    
    u_err = np.sqrt(np.square(( A_G[0,0]*cosa*cosd+A_G[0,1]*sina*cosd+A_G[0,2]*sind))*vec1_err2+\
                np.square((-A_G[0,0]*sina     +A_G[0,1]*cosa                   ))*vec2_err2+\
                np.square((-A_G[0,0]*cosa*sind-A_G[0,1]*sina*sind+A_G[0,2]*cosd))*vec3_err2)
    v_err = np.sqrt(np.square(( A_G[1,0]*cosa*cosd+A_G[1,1]*sina*cosd+A_G[1,2]*sind))*vec1_err2+\
                np.square((-A_G[1,0]*sina     +A_G[1,1]*cosa                   ))*vec2_err2+\
                np.square((-A_G[1,0]*cosa*sind-A_G[1,1]*sina*sind+A_G[1,2]*cosd))*vec3_err2)
    w_err = np.sqrt(np.square(( A_G[2,0]*cosa*cosd+A_G[2,1]*sina*cosd+A_G[2,2]*sind))*vec1_err2+\
                np.square((-A_G[2,0]*sina     +A_G[2,1]*cosa                   ))*vec2_err2+\
                np.square((-A_G[2,0]*cosa*sind-A_G[2,1]*sina*sind+A_G[2,2]*cosd))*vec3_err2)
   
    #lsr_vel=[-8.5,13.38,6.49]
    #lsr_vel=[7.01, 235., 4.95]
    # LSR from Schoenrich et al. 2010, 2012, #Rotation from Irrgang et al.   <-- Uli
    lsr_vel = [11.1, 12.24, 7.25]
    Vlsr = 238.
    #Vlsr = 235
    #lsr_vel = [0,0,0]
    if lsr: 
       u = u+lsr_vel[0]
       v = v+lsr_vel[1]+Vlsr
       w = w+lsr_vel[2]

    V = np.sqrt(u**2 + v**2 +w**2)
    V_err = np.sqrt(np.square(u*u_err)+np.square(v*v_err)+np.square(w*w_err))/V
    return u, v, w, V, u_err, v_err, w_err, V_err

def get_vrf(ra=None, dec=None,
            pmra = None, pmra_err = None,
            pmdec=None, pmdec_err = None,
            vrad = None, vrad_err = None,
            plx  = None, plx_err = None):
    u, v, w, V, u_err, v_err, w_err, V_err = gal_uvw(dis = None, lsr = True, ra=ra, dec=dec,
                                                     pmra = pmra, pmra_err = pmra_err,
                                                     pmdec=pmdec, pmdec_err= pmdec_err,
                                                     vrad = vrad, vrad_err = vrad_err,
                                                     plx  = plx, plx_err=plx_err)
    c = SkyCoord(ra, dec, frame=FK5, unit='deg', equinox='J2015.5')
    xyz = c.galactic.cartesian.xyz.value
    uvw = np.array([u,v,w])
    vrf = np.dot(xyz, uvw)
    vrf_err = np.sqrt((xyz[0]*u_err)**2 +\
                      (xyz[1]*v_err)**2 +\
                      (xyz[2]*u_err)**2)
    return vrf, vrf_err



def get_vrho(dis = None, ra=None, dec=None,
            pmra = None, pmra_err = None,
            pmdec=None, pmdec_err = None,
            vrad = None, vrad_err = None,
            plx  = None, plx_err = None):
    #get RR(kpc), rho(kpc), v_rho(km/s),th(km/s) 
    U, V, W, v, U_err, V_err, W_err, v_err = gal_uvw(dis = dis, lsr = True, ra=ra, dec=dec,
                                                     pmra = pmra, pmra_err = pmra_err,
                                                     pmdec=pmdec, pmdec_err= pmdec_err,
                                                     vrad = vrad, vrad_err = vrad_err,
                                                     plx  = plx,  plx_err=plx_err)
    galcen_distance = 8.27*units.kpc  #Schoenrich et al. 2010, 2012
    #galcen_distance = 8.*units.kpc  #Schoenrich et al. 2010, 2012
    z_sun = 0.0*units.pc
    roll = 0*units.deg
    if np.all(dis) == None:
       dis = 1./plx
    else: dis = dis * 1.e-3
    #c1 = SkyCoord(ra, dec, frame=FK5, unit='deg', equinox='J2015.5')
    #cc1 = c1.transform_to(FK5(equinox='J2000.0'))
    #raJ2000, decJ2000 = cc1.ra.value, cc1.dec.value
    c = coord.ICRS(ra=ra * units.degree,
                dec=dec * units.degree,
                distance=dis * units.kpc)
    aa = c.transform_to(coord.Galactocentric(galcen_distance=galcen_distance,
                                             z_sun=z_sun,roll=roll))
    X, Y, Z = (aa.cartesian.x.value, aa.cartesian.y.value, aa.cartesian.z.value) 
    rho2 = X**2+Y**2
    rho = np.sqrt(rho2)
    RR = np.sqrt(rho2+Z**2)
    vrho = (U*X + V*Y)/rho
    jz = (X*V - Y*U)
    th = -jz/rho
    
    return X,Y,Z,RR,rho,U,U_err,V,V_err,W,W_err,vrho,th, jz



def dis2rv(dis, ra, dec, vv_rotate=238, vv_sun=None, xx_sun=None):
    '''assuming cloud with a rotation velocity of vv_rotate (km/s) around the Galactic center,
       then calculate the radial velocity observed at solar barycenter.
    parameters:
    ----------------
    dis: [float] kpc
    l: [deg] the galactic longitude
    b: [deg] the galactic latitude
    returns:
    --------------
    rv [float] in km/s, radial velocity
    '''
    if vv_sun is None:
       vv_sun = np.array([11.1, 238.+12.24, 7.25])
    if xx_sun is None:
       xx_sun = np.array([-8.27, 0, 0])
    galcen_distance = -xx_sun[0]*units.kpc  #Schoenrich et al. 2010, 2012
    #galcen_distance = 8.*units.kpc  #Schoenrich et al. 2010, 2012
    z_sun = xx_sun[1]*units.pc
    roll = xx_sun[2]*units.deg                                                                                                                                                       
    c = coord.ICRS(ra=ra * units.degree,
                dec=dec * units.degree,
                distance=dis * units.kpc)
    aa = c.transform_to(coord.Galactocentric(galcen_distance=galcen_distance,
                                             z_sun=z_sun,roll=roll))
    x, y, z = (aa.cartesian.x.value, aa.cartesian.y.value, aa.cartesian.z.value)
    #print(x,y, z)
    theta = np.arctan2(y, x) - 0.5*np.pi
    #print(np.rad2deg(theta))
    dvx = vv_rotate*np.cos(theta) - vv_sun[0]
    dvy = vv_rotate*np.sin(theta) - vv_sun[1]
    #print('dv', dvx, dvy)
    dvz = 0 - vv_sun[2]
    dx, dy, dz = x-xx_sun[0], y-xx_sun[1], 0-xx_sun[2]
    #print('dx', dx, dy, dz)
    xx = np.sqrt(dx**2 + dy**2 + dz**2)
    vv = np.sqrt(dvx**2 + dvy**2 + dvz**2)
    #print(vv)
    rv = (dx*dvx + dy*dvy + dz*dvz)/xx
    return rv




if __name__ == '__main__':
   from ten import *
   import astropy.coordinates as coord
   SkyCoord = coord.SkyCoord
   RADEG = 180./np.pi
   ra = ten(8, 44, 47.0)*15.
   dec = ten(11,39, 10.0)
   ra, dec = 255.247021155,7.7410423769
   pmra, pmra_err = -155.154, 1.098
   pmdec, pmdec_err = 8.459, 1.174
   rv, rv_e = -430.0, 9.7
   plx, plx_err = 3.08, 0.43
   u,v,w,V,u_err, v_err, w_err, V_err = gal_uvw (dis = None, lsr = True, ra=ra, dec=dec,
                                                 pmra = pmra, pmra_err = pmra_err,
                                                 pmdec=pmdec, pmdec_err = pmdec_err,
                                                 vrad = rv, vrad_err = rv_e,
                                                 plx = plx, plx_err = plx_err)


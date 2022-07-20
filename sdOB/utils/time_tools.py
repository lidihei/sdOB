from astropy.time import Time
from PyAstronomy import pyasl
from astropy import coordinates as coord
from astropy import units as u
def eval_ltt(ra=62.794724868, dec=50.7082235439, jd=2456326.4583333, site=None, kind='barycentric', barycorr=True):                                                                                                            
    """ evaluate the jd 
    parameters
    ----------
    ra, dec: the coordinate of object
    jd: the julian date of observation (UTC time)
    site: the site of observatory
    return
    ------
    jd_llt: the bjd or hjd
    ltt: light_travel_time
    if barycorr=True return the barycentric correction
    # conf: https://docs.astropy.org/en/stable/time/
    # defaut site is Xinglong
    # coord.EarthLocation.from_geodetic(lat=26.6951*u.deg, lon=100.03*u.deg, height=3200*u.m) lijiang
    """
    if site is None:
        site = coord.EarthLocation.of_site('Beijing Xinglong Observatory')
    # sky position
    ip_peg = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    # time
    times = Time(jd, format='jd', scale='utc', location=site)
    # evaluate ltt
    ltt = times.light_travel_time(ip_peg,kind)
    jd_llt = times.utc + ltt
    if barycorr is True:
       _barycorr = ip_peg.radial_velocity_correction(obstime=Time(times.iso), location=site)
       return jd_llt.jd, ltt, _barycorr
    else:
       return jd_llt.jd, ltt


def eval_ltt_lamost(ra, dec, date_start, date_end, site=None,  local2utc=-8, kind='barycentric', barycorr=True):
    """ evaluate the bjd for lamost
    parameters
    ----------
    ra, dec: the coordinate of object
    date_start: [isot] The observation start local time, header['DATE-BEG'], e.g. '2015-10-31T01:43:45.0'
    date_end: [isot] The observation end local time, header['DATE-BEG'], e.g. '2015-10-31T02:20:32.0'
    site: the site of observatory
    local2utc: [float] (in units of hour) the differece between local time to utc time, default is -8 for Beijing time
    return
    ------
    jd_llt: the bjd or hjd
    ltt: light_travel_time
    if barycorr=True return the barycentric correction
    # conf: https://docs.astropy.org/en/stable/time/
    # defaut site is Xinglong
    # coord.EarthLocation.from_geodetic(lat=26.6951*u.deg, lon=100.03*u.deg, height=3200*u.m) lijiang
    """
    if site is None:
        site = coord.EarthLocation.of_site('Beijing Xinglong Observatory')
    # sky position
    ip_peg = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    # time
    _time = Time([date_start, date_end], format='isot', scale='utc')
    _time = _time + local2utc/24.
    jd = np.mean(_time.jd)
    times = Time(jd, format='jd', scale='utc', location=site)
    # evaluate ltt
    ltt = times.light_travel_time(ip_peg,kind)
    jd_llt = times.utc + ltt
    if barycorr is True:
       _barycorr = ip_peg.radial_velocity_correction(obstime=Time(times.iso), location=site)
       return jd_llt.jd, ltt, _barycorr
    else:
       return jd_llt.jd, ltt

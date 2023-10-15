"""

FUNCTIONS TO CALCULATE BIAS AND CORRELATION FUNCTION 
(2-HALO TERM)

"""

from __future__ import print_function
from __future__ import division

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splev


def wf(y):
    """
    Tophat window function
    """
    w = 3.0*(np.sin(y) - y*np.cos(y))/y**3.0
    if (y>100.0):
        w = 0.0
    return w


def siny_ov_y(y):
    s = np.sin(y)/y
    if (y>100):
        s = 0.0
    return s

def growth_factor(a, cosmo):
    """
    Growth factor from Longair textbook (Eq. 11.56)
    :param cosmo: A astropy cosmo to calculate the Hubble constant
    """
    Om = 0.27+0.0493
    itd = lambda aa: 1.0/(aa*cosmo.H(1.0/aa - 1.0).value)**3.0
    itl = quad(itd, 0.0, a, epsrel=5e-3, limit=100)
    return cosmo.H(1.0/a - 1.0).value*(5.0*Om/2.0)*itl[0]


def bias(var,dcz):
    """
    bias function from Cooray&Sheth Eq.68
    """
    q  = 0.707
    p  = 0.3
    nu = dcz**2.0/var
    e1 = (q*nu - 1.0)/dcz
    E1 = 2.0*p/dcz/(1.0 + (q*nu)**p)
    b1 = 1.0 + e1 + E1
    return b1


def variance(r,TF_tck,Anorm,param):
    """
    variance of density perturbations at z=0
    """
    ns = param.cosmo.ns
    kmin = param.code.kmin
    kmax = param.code.kmax
    itd = lambda logk: np.exp((3.0+ns)*logk) * splev(np.exp(logk),TF_tck)**2.0 * wf(np.exp(logk)*r)**2.0
    itl = quad(itd, np.log(kmin), np.log(kmax), epsrel=5e-3, limit=100)
    var = Anorm*itl[0]/(2.0*np.pi**2.0)
    return var


def correlation(r,TF_tck,Anorm,param):
    """
    Correlation function at z=0
    """
    ns = param.cosmo.ns
    kmin = param.code.kmin
    kmax = param.code.kmax
    itd = lambda logk: np.exp((3.0+ns)*logk) * splev(np.exp(logk),TF_tck)**2.0 * siny_ov_y(np.exp(logk)*r)
    itl = quad(itd, np.log(kmin), np.log(kmax), epsrel=5e-3, limit=100)
    corr = Anorm*itl[0]/(2.0*np.pi**2.0)
    return corr



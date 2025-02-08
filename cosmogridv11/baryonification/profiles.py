"""

PROFILES AND FRACTIONS FOR BARIONIC CORRECTIONS

"""
from __future__ import print_function
from __future__ import division

from scipy.special import erf
from scipy.integrate import simpson, cumulative_trapezoid
from scipy.optimize import fsolve
from scipy.interpolate import splrep, splev
from .constants import *
from .cosmo import *

"""
GENERAL FUNCTIONS REALTED TO THE NFW PROFILE
"""

def r500_fct(r200,c):
    """
    From r200 to r500 assuming a NFW profile
    """
    f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 5.0/2.0*(np.log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return y0*r200


def rvir_fct(r200,c):
    """
    From r500 to r200 assuming a NFW profile
    """
    f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 96.0/200.0*(np.log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return y0*r200


def M500_fct(M200,c):
    """
    From M200 to M500 assuming a NFW profiles
    """
    f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 5.0/2.0*(np.log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return 5.0/2.0*M200*y0**3.0


def cvir_fct(mvir,z):
    """
    Concentrations form Dutton+Maccio (2014)
    c200 (200 times RHOC)
    Assumes PLANCK cosmology
    """
    A = 0.520 + (0.905-0.520)*np.exp(-0.617*z**1.21)
    B = -0.101 + 0.026*z
    return 10.0**A*(mvir/1.0e12)**(B)


"""
STELLAR FRACTIONS
"""

def fSTAR_fct(Mvir,eta=0.3):
    NN = 0.09
    M1 = 2.5e11
    return NN/(Mvir/M1)**(eta)



"""
Generalised (truncated) NFW profiles
"""

def uNFWtr_fct(r,c,t,Mvir,rhoc):
    """
    Truncated NFW density profile. Normalised.
    """
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc))**(1.0/3.0)
    return 1.0/((c*r/rvir)*(1.0+c*r/rvir)**2.0*(1.0+(c*r/rvir)**2.0/t**2.0)**2.0)


def rhoNFW_fct(r,c,Mvir,rhoc):
    """
    NFW density profile.
    """
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc))**(1.0/3.0)
    rho0 = DELTAVIR*rhoc*c**3.0/(3.0*np.log(1.0+c)-3.0*c/(1.0+c))
    return rho0/((c*r/rvir)*(1.0+c*r/rvir)**2.0)

def mNFWtr_fct(x,t):
    """
    Truncated NFW mass profile. Normalised.
    """
    pref   = t**2.0/(1.0+t**2.0)**3.0/2.0
    first  = x/((1.0+x)*(t**2.0+x**2.0))*(x-2.0*t**6.0+t**4.0*x*(1.0-3.0*x)+x**2.0+2.0*t**2.0*(1.0+x-x**2.0))
    second = t*((6.0*t**2.0-2.0)*np.arctan(x/t)+t*(t**2.0-3.0)*np.log(t**2.0*(1.0+x)**2.0/(t**2.0+x**2.0)))
    return pref*(first+second)

def mNFW_fct(x):
    """
    NFW mass profile. Normalised.
    """
    return (np.log(1.0+x)-x/(1.0+x))

def mTOTtr_fct(t):
    """
    Normalised total mass (from truncated NFW)
    """
    pref   = t**2.0/(1.0+t**2.0)**3.0/2.0
    first  = (3.0*t**2.0-1.0)*(np.pi*t-t**2.0-1.0)
    second = 2.0*t**2.0*(t**2.0-3.0)*np.log(t)
    return pref*(first+second)

def mTOT_fct(c):
    """
    Normalised total mass (from NFW)
    """
    return np.log(1.0 + c) - c/(1.0 + c)


def MNFWtr_fct(r,c,t,Mvir,rhoc):
    """
    Truncateed NFW mass profile.
    """
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc))**(1.0/3.0)
    return Mvir*mNFWtr_fct(c*r/rvir,t)/mNFWtr_fct(c,t)

def MNFW_fct(r,c,Mvir,rhoc):
    """
    NFW mass profile
    """
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc))**(1.0/3.0)
    return (np.log(1.0+c*r/rvir)-c*r/rvir/(1.0+c*r/rvir))/(np.log(1.0+c)-c/(1.0+c))*Mvir


"""
GAS PROFILE
"""

def beta_fct(Mv,Mc,c,mu):
    """
    Parametrises slope of gas profile
    """
    beta = c-(Mc/Mv)**mu
    if (beta<-10.0):
        beta = -10.0
    return beta


def uHGA_fct(r,rm,rc,al,be,ga):
    """
    Normalised gas density profile
    """
    x = r/rc
    y = r/rm
    return 1.0/(1.0+x**al)**(be)/(1.0+y**ga)**((7.0-al*be)/ga)


"""
STELLAR PROFILE
"""

def uCGA_fct(r,Mvir,rhoc):
    """
    Normalised density profile of central galaxy
    """
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc))**(1.0/3.0)
    R12  = 0.015*rvir
    return np.exp(-(r/R12/2.0)**2.0)/r**2.0


def MCGA_fct(r,Mvir,rhoc):
    """
    Normalised mass profile of central galaxy
    (needs to be multiplied with Mtot)
    """
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc))**(1.0/3.0)
    R12  = 0.015*rvir
    return erf(r/R12/2.0)


"""
TOTAL PROFILE
"""

# @profile
def profiles(rbin, Mvir, cvir, Mc, mu, thej, cosmo_corr, cosmo_bias, param, rhoc, tr_cvir, tr_tau, tr_Mvir, adiabatic_exp_contr='Abadi2010'):
    """
    Calculates fractions, density and mass profiles as a function of radius
    Returns a dictionary
    """
    # if the truncated parameters are not defined we need to set them
    if tr_cvir < 0:
        tr_cvir = cvir
        tr_tau = param.code.eps*cvir
        tr_Mvir = Mvir

    #parameters
    thco    = param.baryon.thco
    eta_tot = param.baryon.eta_tot
    eta_cga = param.baryon.eta_cga
    Om      = param.cosmo.Om
    Ob      = param.cosmo.Ob
    #radii
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc))**(1.0/3.0)
    rej  = thej*rvir
    rco  = thco*rvir

    #total fractions
    fbar  = Ob/Om
    fcdm  = (Om-Ob)/Om
    fstar = fSTAR_fct(Mvir,eta_tot)
    fcga  = fSTAR_fct(Mvir,eta_cga) #Moster13
    fsga  = fstar-fcga #satellites and intracluster light
    if(fsga<0):
        print('ERROR: negative fraction of satellite galaxies')
        exit()
    fhga  = fbar-fcga-fsga

    #total dark-matter-only mass, CHECKED THAT IT WORKS FOR TRNFW FIT
    Mtot = tr_Mvir*mTOTtr_fct(tr_tau)/mNFWtr_fct(tr_cvir,tr_tau)

    #Initial density and mass profiles
    rho0NFWtr = DELTAVIR*rhoc*tr_cvir**3.0/(3.0*mNFWtr_fct(tr_cvir,tr_tau))
    rhoNFW    = rho0NFWtr*uNFWtr_fct(rbin,tr_cvir,tr_tau,tr_Mvir,rhoc)
    rho2h     = (cosmo_bias*cosmo_corr + 1.0)*Om*RHOC #rho_b=const in comoving coord.
    rhoDMO    = rhoNFW + rho2h
    MNFW      = MNFWtr_fct(rbin,tr_cvir,tr_tau,tr_Mvir,rhoc)
    M2h       = cumulative_trapezoid(4.0*np.pi*rbin**2.0*rho2h,rbin,initial=rbin[0])
    MDMO      = MNFW + M2h

    #Final density and mass profiles
    alpha = 1.0
    beta  = beta_fct(Mvir,Mc,3.0,mu)
    gamma = 2.0
    uHGA     =  uHGA_fct(r=rbin,rm=rej,rc=rco,al=alpha,be=beta,ga=gamma)
    rho0HGA  = Mtot/(4.0*np.pi*simpson(rbin**2.0*uHGA,x=rbin))
    rhoHGA   = rho0HGA*uHGA
    R12      = 0.015*rvir
    rho0CGA  = Mtot/(4.0*np.pi**(3.0/2.0)*R12)
    rhoCGA   = rho0CGA*uCGA_fct(rbin,Mvir,rhoc)
    MHGA     = cumulative_trapezoid(4.0*np.pi*rbin**2.0*rhoHGA,rbin,initial=rbin[0]) + M2h
    MCGA     = Mtot*MCGA_fct(rbin,Mvir,rhoc) + M2h
    MHGA_tck = splrep(rbin, MHGA, s=0, k=3)
    MCGA_tck = splrep(rbin, MCGA, s=0, k=3)

    #Adiabatic contraction/expansion (after Abadi et al 2010)
    if adiabatic_exp_contr == 'Abadi2010':
        
        MNFWri = MDMO
        aa = 0.3
        nn = 2.0
        func = lambda x: (x-1.0) - aa*((MNFWri/((fcdm+fsga)*MNFWri + fcga*splev(x*rbin,MCGA_tck,der=0) + fhga*splev(x*rbin,MHGA_tck,der=0)))**nn - 1.0)
    
    #Adiabatic contraction/expansion (Gnedin 2004, see also Teyssier et al 2011)
    elif adiabatic_exp_contr == 'Gnedin2004':

        MNFWri = MDMO
        aa = 0.68
        func = lambda x: (x-1.0) - aa*(MNFWri/((fcdm+fsga)*MNFWri + fcga*splev(x*rbin,MCGA_tck,der=0) + fhga*splev(x*rbin,MHGA_tck,der=0)) - 1.0)

    else:
        raise Exception(f'unknown Adiabatic contraction/expansion {adiabatic_exp_contr}, choose from [Abadi2010, Gnedin2004]')

    xi = np.empty(len(rbin)); xi.fill(1.0)
    xx = fsolve(func, xi, fprime=None, band=(0,0))

    MACM     = MNFWtr_fct(rbin/xx,tr_cvir,tr_tau,Mvir,rhoc)
    MACM_tck = splrep(rbin, MACM, s=0, k=3)
    rhoACM   = splev(rbin,MACM_tck,der=1)/(4.0*np.pi*rbin**2.0)
    MACM     = MACM + M2h

    #total profile
    rhoBAR   = (fcdm+fsga)*rhoACM + fhga*rhoHGA + fcga*rhoCGA
    rhoDMB   = rhoBAR + rho2h
    MDMB     = (fcdm+fsga)*MACM + fhga*MHGA + fcga*MCGA

    # grid for integration
    # the axis we integrate out should be fixed across all halos
    r_int = rbin
    r_int = np.sqrt(rbin[None, :] ** 2 + r_int[:, None] ** 2)

    # get the spline for the normal densities
    tck_DMO = splrep(rbin, rhoDMO, s=0, k=3)
    tck_DMB = splrep(rbin, rhoDMB, s=0, k=3)
    rhoDMO_int = splev(tck=tck_DMO, x=r_int, ext=1)
    rhoDMB_int = splev(tck=tck_DMB, x=r_int, ext=1)

    # integrate (factor of 2 because r_int > 0, we integrate from the plane through the center)
    projected_rho_DMO = 2*np.trapz(rhoDMO_int, rbin, axis=0)
    projected_rho_DMB = 2*np.trapz(rhoDMB_int, rbin, axis=0)
    integrated_mass_DMO = cumulative_trapezoid(2.0 * np.pi * rbin * projected_rho_DMO, rbin, initial=1e-8)
    integrated_mass_DMB = cumulative_trapezoid(2.0 * np.pi * rbin * projected_rho_DMB, rbin, initial=1e-8)

    #define dictionaries
    frac = { 'CDM':fcdm, 'CGA':fcga, 'SGA':fsga, 'HGA':fhga }
    dens = { 'NFW':rhoNFW, 'BG':rho2h, 'DMO':rhoDMO, 'ACM':rhoACM, 'CGA':rhoCGA, 'HGA':rhoHGA, 'DMB':rhoDMB }
    mass = { 'NFW':MNFW, 'BG':M2h, 'DMO':MDMO, 'ACM':(fcdm+fsga)*MACM, 'CGA':fcga*MCGA, 'HGA':fhga*MHGA, 'DMB':MDMB, 'MtotDMO': Mtot, 'DMO_pro': integrated_mass_DMO, 'DMB_pro': integrated_mass_DMB}
    return frac, dens, mass






# code graveyard




    # # user specified fast_mode or rbin is not an array
    # if fast_mode and (not isinstance(rbin, float)):

    #     delta = np.expand_dims(rbin, axis=0)/np.expand_dims(rbin, axis=1)   
    #     f = (delta-1.0) - aa*((MNFWri/((fcdm+fsga)*MNFWri + fcga*MCGA + fhga*MHGA))**nn - 1.0)
    #     # where there is no solution (f>0), this will take the 0-th element
    #     ix = np.argmax(np.where(f<0, f, -1e20), axis=1, keepdims=True)

    #     # linear interp
    #     ix = np.concatenate([ix, ix+1], axis=1)
    #     delta_ix = np.take_along_axis(delta, ix, axis=1)
    #     f_delta = np.take_along_axis(f, ix, axis=1)
    #     a_fit = (f_delta[:,0]-f_delta[:,1])/(delta_ix[:,0]-delta_ix[:,1])
    #     b_fit = f_delta[:,0]-a_fit*delta_ix[:,0]
    #     xx = -b_fit/a_fit
    #     # np.max(np.abs(xx3.ravel()-xx.ravel()))

    # else:
    
    
    # def func(x):

    #     # func = lambda x: (x-1.0) - aa*((MNFWri/((fcdm+fsga)*MNFWri + fcga*splev(x*rbin,MCGA_tck,der=0) + fhga*splev(x*rbin,MHGA_tck,der=0)))**nn - 1.0)
    #     sp_mcga = splev(x*rbin,MCGA_tck,der=0)
    #     sp_mhga = splev(x*rbin,MHGA_tck,der=0)
    #     out = (x-1.0) - aa*((MNFWri/((fcdm+fsga)*MNFWri + fcga*sp_mcga + fhga*sp_mhga))**nn - 1.0)
    #     return out

    # def func2(x, rbin, MCGA, MHGA, fcga, fhga, fcdm, fsga, MNFWri, nn, aa):

    #     # func = lambda x: (x-1.0) - aa*((MNFWri/((fcdm+fsga)*MNFWri + fcga*splev(x*rbin,MCGA_tck,der=0) + fhga*splev(x*rbin,MHGA_tck,der=0)))**nn - 1.0)
    #     # sp_mcga = splev(x*rbin,MCGA_tck,der=0)
    #     # sp_mhga = splev(x*rbin,MHGA_tck,der=0)

    #     sp_mcga = np.interp(x*rbin,rbin,MCGA)
    #     sp_mhga = np.interp(x*rbin,rbin,MHGA)

    #     out = (x-1.0) - aa*((MNFWri/((fcdm+fsga)*MNFWri + fcga*sp_mcga + fhga*sp_mhga))**nn - 1.0)
    #     return out

    # def func3(x, rbin, MCGA_tck, MHGA_tck, fcga, fhga, fcdm, fsga, MNFWri, nn, aa):

    #     # func = lambda x: (x-1.0) - aa*((MNFWri/((fcdm+fsga)*MNFWri + fcga*splev(x*rbin,MCGA_tck,der=0) + fhga*splev(x*rbin,MHGA_tck,der=0)))**nn - 1.0)
    #     sp_mcga = splev(x*rbin,MCGA_tck,der=0)
    #     sp_mhga = splev(x*rbin,MHGA_tck,der=0)

    #     out = (x-1.0) - aa*((MNFWri/((fcdm+fsga)*MNFWri + fcga*sp_mcga + fhga*sp_mhga))**nn - 1.0)
    #     return out
    
    # def get_xx_fast():
    
    #     delta = np.expand_dims(rbin, axis=0)/np.expand_dims(rbin, axis=1)
    #     f = (delta-1.0) - aa*((MNFWri/((fcdm+fsga)*MNFWri + fcga*MCGA + fhga*MHGA))**nn - 1.0)
    #     ix = np.argmax(np.where(f<0, f, -1e20), axis=1, keepdims=True)
    #     xx1 = np.take_along_axis(delta, ix, axis=1).ravel()
    #     xx2 = np.take_along_axis(delta, ix+1, axis=1).ravel()
    #     f1 = np.take_along_axis(f, ix, axis=1).ravel()
    #     f2 = np.take_along_axis(f, ix+1, axis=1).ravel()

    #     # linear interp
    #     a_fit = (f1-f2)/(xx1-xx2)
    #     b_fit = f1-a_fit*xx1
    #     xx3 = -b_fit/a_fit
    #     # np.max(np.abs(xx3.ravel()-xx.ravel()))

    #     return xx3

    # func2p = partial(func2, x=xi, rbin=rbin, MCGA=MCGA, MHGA=MHGA, fcga=fcga, fhga=fhga, fcdm=fcdm, fsga=fsga, MNFWri=MNFWri, nn=nn, aa=aa)
    # timeit.timeit(func2p, number=1000)/1000

    # func2p = partial(func2, rbin=rbin, MCGA=MCGA, MHGA=MHGA, fcga=fcga, fhga=fhga, fcdm=fcdm, fsga=fsga, MNFWri=MNFWri, nn=nn, aa=aa)
    # ftt = partial(bisect_vec, func=func2p, a=np.full(len(xi), 0.), b=np.full(len(xi), 2.), tol=1e-8)
    # timeit.timeit(ftt, number=1000)/1000

    # ftt = partial(fsolve, func=func2p, x0=xi, band=(0,0))
    # timeit.timeit(ftt, number=1000)/1000

    # func3p = partial(func3, rbin=rbin, MCGA_tck=MCGA_tck, MHGA_tck=MHGA_tck, fcga=fcga, fhga=fhga, fcdm=fcdm, fsga=fsga, MNFWri=MNFWri, nn=nn, aa=aa)
    # ftt = partial(bisect_vec, func=func3p, a=np.full(len(xi), 0.), b=np.full(len(xi), 2.), tol=1e-8)
    # print(timeit.timeit(ftt, number=1000)/1000)
    # ftt = partial(fsolve, func=func3p, x0=xi)
    # print(timeit.timeit(ftt, number=1000)/1000)

    # func = lambda x: (x-1.0) - aa*((MNFWri/((fcdm+fsga)*MNFWri + fcga*splev(x*rbin,MCGA_tck,der=0) + fhga*splev(x*rbin,MHGA_tck,der=0)))**nn - 1.0)
    # func_val = aa*((MNFWri/((fcdm+fsga)*MNFWri + fcga*splev(rbin,MCGA_tck,der=0) + fhga*splev(rbin,MHGA_tck,der=0)))**nn - 1.0)
    # spl_func_tck = splrep(rbin, func_val, s=0, k=3)
    # loss = lambda x, r: (x-1 - splev(r*x, spl_func_tck, der=0))**2
    # from scipy.optimize import minimize
    # minimize(loss, x0=xi, args=(rbin[0]))

    # import pudb; pudb.set_trace();
    # pass

    # xx_bisect = bisect_vec(func, a=np.full(len(xi), 0.), b=np.full(len(xi), 2.), tol=1e-8)
    # xx_old = fsolve(func3p, xi, fprime=None, band=(0,0))

    # xx = t[0]
    # xx = fsolve(func, xi, fprime=None, band=(0,0), xtol=1e-8)
    # xx2 = get_xx_fast()
    # xx = fsolve(func, xi, fprime=None, band=(0,0), xtol=1e-8)
    # xx = fsolve(func, xi, fprime=None)
    # xx, infodict, ier, mesg = fsolve(func, xi, fprime=None, band=(0,0), xtol=1e-8, full_output=True)

    # from functools import partial
    # from timeit import timeit
    # timeit(partial(fsolve, func=func, x0=xi, xtol=1e-8,  band=(0,0)), number=1000)/1000
    # timeit(partial(bisect_vec, func=func, a=np.full(len(xi), 0.), b=np.full(len(xi), 2.), tol=1e-8), number=1000)/1000

    # for x in np.linspace(0.5, 1.5):
    #     func(x)

    # if fast_mode:
        
    #     rhoDMO_int = np.interp(r_int, rbin, rhoDMO, left=0, right=0)
    #     rhoDMB_int = np.interp(r_int, rbin, rhoDMB, left=0, right=0)


# def bisect_vec(func, a, b, n_iter_max=1000, tol=1e-8):

#     fa = func(a)
#     fb = func(b)
    
#     for i in range(n_iter_max):

#         p = (a+b)/2
#         fp = func(p)
#         mb = fa*fp<0
#         ma = ~mb
#         np.copyto(a, p, casting='no', where=ma)
#         np.copyto(b, p, casting='no', where=mb)
#         np.copyto(fa, fp, casting='no', where=ma)
#         np.copyto(fb, fp, casting='no', where=mb)

#         loss = np.mean(np.abs(fp))

#         if loss<tol:
#             break

#     return p, i

"""
External Parameters
"""

class Bunch(object):
    """
    translates dic['name'] into dic.name 
    """

    def __init__(self, data):
        self.__dict__.update(data)

def cosmo_par():
    par = {
        "Om": 0.315,
        "Ob": 0.049,
        "s8": 0.83,
        "h0": 0.673,
        "ns": 0.963,
        "dc": 1.675,
        "w0": -1.0,
        "m_nu": 0.02, # neutrino mass (None for massless neutrinos)
        }
    return Bunch(par)

def baryon_par():
    par = {
        "Mc": 3.0e13,     #beta_fct(M): critical mass scale
        "nu": 0.0,        #redshift dependence of Mc: Mc(z) = Mc*(1+z)**nu
        "mu": 0.3,        #beta_fct(M): critical mass scale
        "thej": 4.0,      #ejection factor thej=rej/rvir
        "thco": 0.1,      #core factor thco=rco/rvir
        "eta_tot": 0.32,  #Mstar/Mvir~(10**11.435/Mvir)**-eta_tot (tot = cga + satelleites)
        "eta_cga": 0.60,  #Mstar/Mvir~(10**11.435/Mvir)**-eta_cga (cga = central galaxy)
        }
    return Bunch(par)

def io_files():
    par = {
        "transfct": 'CDM_PLANCK_tk.dat',
        "transfct_format": 'CAMB', # CAMB (text file) or HDF5 (concept hdf5 file)
        "halofile_format": 'AHF-ASCII',
        "shell_format": 'FITS',
    }
    return Bunch(par)

def code_par():
    par = {
        "kmin": 0.01,
        "kmax": 100.0,
        "rmin": 0.005,
        "rmax": 50.0,
        # max number of viral radius to take (rmax in case maxRvir*rvir > rmax)
        "maxRvir": 20,
        # truncation factor: eps=rtr/rvir
        "eps": 4.0,
        # mode of the pkd halo fitting
        "mode": "VPNFW",
        # halo part counting
        "count": "vir",
        # truncate viral profiles
        "VP_truncate": True,
        # force halos with no particles to have at least the four neighbouring pixels
        "force_parts": False,
        # halos that are close to a shell boundary will be part of both shells
        "halo_buffer": -1,
        }
    return Bunch(par)

def sim_par():
    par = {
        "nparts": 128, # number of particles in the simulation per side
        "Lbox": 128.0,   #box size of partfile_in [Mpc/h]
        "Nmin_per_halo": 100,
        }
    return Bunch(par)

def par():
    par = Bunch({"cosmo": cosmo_par(),
        "baryon": baryon_par(),
        "files": io_files(),
        "code": code_par(),
        "sim": sim_par(),
        })
    return par

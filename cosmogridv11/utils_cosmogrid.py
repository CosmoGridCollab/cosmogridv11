import os, sys, shutil, stat, logging, subprocess, shlex, collections, datetime, numpy as np, pickle, importlib, h5py
from . import utils_logging, utils_config, utils_io
LOGGER = utils_logging.get_logger(__file__)

set_types = ['all', 'grid', 'fiducial']

def get_simulations_list(set_type='all'):
    """
    :param sim_set: which type of sims to load [all, grid, fiducial]
    :return simslist_fiducial: simulations list
    :return parslist_fiducial: input parameters list
    """


    # get simulation list
    dir_resources = os.path.join(os.path.dirname(__file__), 'resources')

    metainfo = utils_io.read_from_pickle(os.path.join(dir_resources, 'CosmoGrid_metainfo.pkl.gz'))

    simslist = metainfo[f'simulations/{set_type}']
    parslist = metainfo[f'parameters/{set_type}']
    shell_info = {x[0]:x[1] for x in filter(lambda x: x[0] in metainfo[f'parameters/{set_type}']['path_par'], metainfo['shell_info'].items())}

    return simslist, parslist, shell_info


def build_cosmo(params):

    from astropy.cosmology import FlatwCDM
    from astropy.units import eV
    
    TCMB0 = 2.7255 # cmb temperature
    NEFF = 3.046 # effective number of neutrino species

    astropy_cosmo = FlatwCDM(Om0=params['O_cdm'] + params['Ob'], 
                             H0=params['H0'], 
                             Tcmb0=TCMB0,
                             Neff=NEFF, 
                             m_nu=params['m_nu']*eV, 
                             Ob0=params['Ob'], 
                             w0=params['w0'])

    return astropy_cosmo

def parse_par_ids(par_ids, n_max):

    par_groups = par_ids.split(',')
    list_pars = []
    for par_group in par_groups:

        if ':' in par_group:
            p1, p2 = par_group.split(':')
            p1 = 0 if p1 == '' else p1
            p2 = int(n_max) if p2 == '' else p2
            list_pars += range(int(p1), int(p2))
        else:
            list_pars += [int(par_group)]

    list_pars = np.sort(np.unique(list_pars))
    LOGGER.info(f'parsed param_ids={par_ids}, n_pars={len(list_pars)}')

    return list_pars


def is_fiducial(sim_params):

    return '/fiducial/' in sim_params['path_par']

def get_sims_path(sim_params, id_sim):

    if is_fiducial(sim_params):

        root_sim = f"run_{id_sim:04d}"

    else:

        root_sim = f"run_{id_sim:d}"

    path_sim = os.path.join(sim_params['path_par'], root_sim)
    return path_sim
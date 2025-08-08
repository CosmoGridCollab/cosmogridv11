import os, sys, shutil, stat, logging, subprocess, shlex, collections, datetime, numpy as np, pickle, importlib, h5py
from . import utils_logging, utils_config, utils_io, utils_arrays
LOGGER = utils_logging.get_logger(__file__)

set_types = ['all', 'grid', 'fiducial']


def load_permutations_list(conf):
    """Summary
    
    Parameters
    ----------
    conf : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    assert 'redshift_perturbations_list' in conf['paths'], 'path to shell permutation list missing, add config[paths][redshift_perturbations_list]'
    filename_permlist = str(conf['paths']['redshift_perturbations_list'])
    permlist = np.load(filename_permlist)
    LOGGER.info(f'loaded {filename_permlist} with {len(permlist)} permutations')
    
    return permlist

def read_cosmogrid_metainfo(fpath):

    def get_dataset_keys(f):
        keys = []
        f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
        return keys

    keys_copy = ['parameters', 'simulations']
    sim_types = ['all', 'fiducial', 'grid', 'benchmark']
    metainfo = {}

    with h5py.File(fpath, 'r') as f:

        for k in keys_copy:
            for s in sim_types:
                field = f'{k}/{s}'
                metainfo[field] = utils_arrays.ascii_to_unicode(np.array(f[field]))

        metainfo['shell_info'] = {}
        for k in get_dataset_keys(f['shell_info']):
            metainfo['shell_info'][k] = np.array(f['shell_info'][k])

    return metainfo




def get_simulations_list(set_type='all'):
    """
    :param sim_set: which type of sims to load [all, grid, fiducial]
    :return simslist_fiducial: simulations list
    :return parslist_fiducial: input parameters list
    """


    # get simulation list
    dir_resources = os.path.join(os.path.dirname(__file__), '..', 'resources')
    fpath = os.path.join(dir_resources, 'CosmoGridV11_metainfo.h5')
    metainfo = read_cosmogrid_metainfo(fpath)
    # metainfo = utils_io.read_from_pickle(os.path.join(dir_resources, 'CosmoGrid_metainfo.pkl.gz'))

    simslist = metainfo[f'simulations/{set_type}']
    parslist = metainfo[f'parameters/{set_type}']
    shell_info = {x[0]:x[1] for x in filter(lambda x: x[0] in metainfo[f'parameters/{set_type}']['path_par'], metainfo['shell_info'].items())}

    LOGGER.info('loaded raw CosmoGrid metainfo table ' + fpath)

    return simslist, parslist, shell_info


def get_baryonified_simulations_list(conf, set_type='all'):
    """
    :param sim_set: which type of sims to load [all, grid, fiducial]
    :return simslist_fiducial: simulations list
    :return parslist_fiducial: input parameters list
    """

    # get the correct path
    conf['paths'].setdefault('metainfo_bary', 'Fluri+2022')
    if conf['paths']['metainfo_bary'] == 'Fluri+2022':

        LOGGER.warning('using CosmoGridV1 default baryonification from Fluri+2022')
        dir_resources = os.path.join(os.path.dirname(__file__), '..','resources')
        conf['paths']['metainfo_bary'] = os.path.join(dir_resources, 'CosmoGridV11_bary_Fluri+2022_metainfo.h5')

    # read table
    metainfo = read_cosmogrid_metainfo(conf['paths']['metainfo_bary'])
    LOGGER.info('loaded baryonified CosmoGrid metainfo table ' + conf['paths']['metainfo_bary'])
    
    # select set
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


def decompress_params_files(filename_params):

    import tarfile
    with tarfile.open(filename_params, "r") as tf:
        tf.extract(member='CosmoML.log')
        tf.extract(member='baryonification_params.py')
        tf.extract(member='class_processed.hdf5')
import os, sys, shutil, stat, logging, subprocess, shlex, collections, datetime, numpy as np, pickle, importlib, h5py
from . import utils_logging, utils_io
LOGGER = utils_logging.get_logger(__file__)


def load_config(filename):

    conf = utils_io.read_yaml(filename)
    if 'redshifts_nz' in conf.keys():

        for v in conf['redshifts_nz']:
            
            if 'file' in v:
                v['file'] = v['file'].replace('$MODULE_DIR', os.path.dirname(__file__))
            
            v.setdefault('z_max', 1e6)

    conf.setdefault('redshift_error_method', 'fischbacher')
    conf.setdefault('projection', {})
    conf['projection'].setdefault('shell_perms', True)
    conf['projection'].setdefault('shell_perms_seed', 424242)
    conf['projection'].setdefault('survey_mask', False)

    if type(conf['projection']['survey_mask']) is str:
        conf['projection']['survey_mask'] = conf['projection']['survey_mask'].replace('$MODULE_DIR', os.path.dirname(__file__))

    return conf


def load_cosmogrid_shell_info(fname, parslist):

    shell_info = {}
    with h5py.File(fname, 'r') as f:

        for par in parslist:

            dset_name = par['path_par'].rstrip('/')
            shell_info[dset_name] = np.sort(np.array(f[dset_name]), order='shell_id')

    LOGGER.info(f'loaded shell_info with {len(shell_info)} parameters from {fname}')
    return shell_info

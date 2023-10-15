# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2022
author: Tomasz Kacprzak
"""

import os, sys, warnings, argparse, h5py, numpy as np, time, logging, itertools, shutil
from cosmogrid_des_y3 import utils_io, utils_logging, utils_config, utils_cosmogrid, utils_arrays, utils_shell_permutation, utils_maps
from cosmogrid_des_y3.filenames import *
import healpy as hp
from UFalcon import probe_weights
from cosmogrid_des_y3.copy_guardian import NoFileException


warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

def setup(args):

    description = 'Make maps'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
                        help='logging level')
    parser.add_argument('--dir_root', type=str, required=True, 
                        help='dir with shell maps (either npz or raw)')
    parser.add_argument('--nside_out', type=int, default=512,
                        help='output nside')
    parser.add_argument('--box_size', type=float, default=900,
                        help='box size in Mpc/h')
    parser.add_argument('--n_particles', type=int, default=832**3,
                        help='number of particles')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    args = parser.parse_args(args)

    utils_logging.set_all_loggers_level(args.verbosity)

    # get absolute paths
    args.dir_root = utils_io.get_abs_path(args.dir_root)


    return args


def resources(args):
    
    res = {'main_memory':24000,
           'main_time_per_index':24/10, # hours
           'main_scratch':16000,
           'merge_memory':64000,
           'merge_time':24}
           # 'pass':{'constraint': 'knl', 'account': 'des', 'qos': 'regular'}} # Cori
    
    return res


def main(indices, args):
    """
    Project shells using probe weights computed realier.
    Code lifted from Janis Fluri's Kids1000 analysis.
    https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/map_projection/patch_generation/project_patches.py#L1
    """

    from cosmogrid_des_y3.copy_guardian import NoFileException

    args = setup(args)

    baryontest_cases = ['shell_baryonification', 'no_baryonification', 'aurel_baryonification']

    for index in indices:

        LOGGER.info(f'running on index {index} {baryontest_cases[index]}')

        # get kernels
        file_weights = get_filename_probe_weights(args.dir_root)
        probe_weights = utils_maps.load_probe_weigths(file_weights)
        probes = list(probe_weights.keys())
        LOGGER.info(f'using probes {probes}')
        
        # get shells    
        dir_case = os.path.join(args.dir_root, baryontest_cases[index])
        path_sim = os.path.join(dir_case, 'shell_baryonification_shells.npz')
        if os.path.isfile(path_sim):
            shells = utils_maps.load_compressed_shells(path_sim)

        else:    
            shells = utils_maps.load_shells_uncompressed(dir_case)
    
        nside_raw = hp.npix2nside(shells.shape[1])
        if nside_raw != args.nside_out:
            scale = hp.nside2npix(nside_raw)/hp.nside2npix(args.nside_out)
            shells = hp.ud_grade(shells, nside_out=args.nside_out)*scale
            
        probe_maps = utils_maps.project_all_probes(shells, probes, probe_weights, 
                                                   nside=args.nside_out, 
                                                   n_particles=args.n_particles, 
                                                   box_size=args.box_size)

        filename_out = get_filepath_projected_maps(dir_case, variant=f'baryontest')
        utils_maps.store_probe_maps(filename_out, probe_maps)

        yield index



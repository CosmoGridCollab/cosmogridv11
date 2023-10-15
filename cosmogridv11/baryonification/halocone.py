# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2022
author: Tomasz Kacprzak
"""

import os, sys, warnings, argparse, h5py, numpy as np, time, logging, itertools, shutil
from cosmogrid_des_y3 import utils_io, utils_logging, utils_config, utils_cosmogrid
from cosmogrid_des_y3.filenames import *
import healpy as hp

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

def extract_redshift_bounds(log_file):

    z_edge = np.genfromtxt(log_file)[:,1]
    lower_z = z_edge[1:]
    upper_z = z_edge[:-1]

    # Load the shells to get the redshift boundaries of each shell
    z_bounds = np.stack([lower_z, upper_z], axis=1)

    return z_bounds
    


def get_halocone(filename_params, filename_halos, z_lims=[0,5], test=False):
    """
    https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/carpet/baryons/gen_halocone.py
    This file creates a HaloCone from a CosmoGrid simulation.
    """
    
    import baryonification


    # initialise parameters
    from baryonification_params import par

    ###############
    # Definitions #
    ###############

    compressed_shells = "compressed_shells.npz"  # file with the compressed shells
    log_file = "CosmoML.log"  # log file of the simulation
    halo_file_prefix = "./pkd_halos/CosmoML"  # prefix for the halofiles, i.e. everything before <step>.fofstat.0
    outpath = "./"  # path for the output HaloCone
    n_proc = 1  # Number of processors to use
    verbosity = 4 # verbosity for the output, set to 10 for max output (which will be tons of output)

    ############
    # The Code #
    ############

    par.sim.Nmin_per_halo = 50
    par.code.mode         = "NFW"
    par.code.count        = "tot"
    par.code.halo_buffer  = 0.0

    # Load the shells to get the redshift boundaries of each shell
    z_bounds = extract_redshift_bounds(log_file)
    print("Extracted redshift bounds: ", z_bounds)

    # Match the corresponding HaloFiles
    halo_files = np.array(baryonification.utils.get_halofiles_from_z_boundary(z_bounds, log_file, halo_file_prefix))
    print("Matched Halofiles: ", halo_files)


    if test:
        LOGGER.warning('=============> test!')
        halo_files = halo_files[:55]
        z_bounds = z_bounds[:55]
    else:
        select = (z_bounds[:,1] > z_lims[0]) & (z_bounds[:,0] < z_lims[1])
        LOGGER.info(f"selected {np.count_nonzero(select)}/{len(select)} halo snapshots for z-range [{z_lims[0]},{z_lims[1]}]")
        halo_files = halo_files[select]
        z_bounds = z_bounds[select]


    # write
    out_file = baryonification.prep_lightcone_halos(par, halo_files, z_bounds, outpath, verbosity=verbosity, n_proc=n_proc, add_obs_coords=True)
    print("Output Halo file: {}".format(out_file))

    halos_container=np.load(f'Halofile_MinParts={par.sim.Nmin_per_halo}.npz')
    halos = np.array(halos_container['halos'])
    shell_info = np.array(halos_container['shells'])

    return halos, shell_info

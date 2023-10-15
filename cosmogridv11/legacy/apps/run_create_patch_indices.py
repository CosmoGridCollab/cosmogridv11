# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2022
author: Arne Thomsen
"""

import healpy as hp
import numpy as np
import h5py
import os
import warnings
import argparse
from functools import reduce

from cosmogrid_des_y3 import utils_logging, utils_io
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

""" 
based of the following notebooks
https://cosmo-gitlab.phys.ethz.ch/arne.thomsen/cosmogrid_desy3/-/blob/main/notebooks/patches.ipynb
https://cosmo-gitlab.phys.ethz.ch/arne.thomsen/cosmogrid_desy3/-/blob/main/notebooks/index_file.ipynb

which are based off Janis' work
https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/carpet/notebooks/KiDS_1000_index_file.ipynb

This app only needs to be ran once to generate the index file
"""

# constants TODO should not be hardcoded like this
nside = 512
npix = hp.nside2npix(nside)
n_z_bins = 5

# continous rotations (on catalog level)

def get_rot_x(ang):
    return np.array([[1.0, 0.0,         0.0],
                     [0.0, np.cos(ang), -np.sin(ang)],
                     [0.0, np.sin(ang), np.cos(ang)]]).T # Inverse because of healpy

def get_rot_y(ang):
    return np.array([[np.cos(ang),  0.0, np.sin(ang)],
                     [0.0,          1.0, 0.0],
                     [-np.sin(ang), 0.0, np.cos(ang)]]).T # Inverse because of healpy

def get_rot_z(ang):
    return np.array([[np.cos(ang), -np.sin(ang), 0.0],
                     [np.sin(ang), np.cos(ang),  0.0],
                     [0.0,         0.0,          1.0]]).T # Inverse because of healpy

# healpy symmetry transforms (on map level)

def mirror_pix(pix, lr=False, nside=nside):
    """
    mirror a patch

    :param pix: patch pixel indices
    :param nside: healpy nside
    :param lr: left to right (instead of up down) mirroring
    """
    theta, phi = hp.pix2ang(ipix=pix, nside=nside)
    if lr:
        phi = 2*np.pi - phi
    else:
        theta = np.pi - theta
    new_pix = hp.ang2pix(theta=theta, phi=phi, nside=nside)

    # we make sure that no pixel appears twice that should not
    assert len(set(new_pix)) == len(set(pix))

    return new_pix

def rotate_pix(pix, n_rot=1, nside=nside):
    """
    rotate a patch by 90 degrees
    
    :param pix: patch pixel indices
    :param n_rot: number of rotations
    :param nside: healpy nside
    """
    theta, phi = hp.pix2ang(ipix=pix, nside=nside)
    phi = (phi + n_rot*np.pi/2) % (2*np.pi)
    new_pix = hp.ang2pix(theta=theta, phi=phi, nside=nside)

    # we make sure that no pixel appears twice that should not
    assert len(set(new_pix)) == len(set(pix))

    return new_pix


def resources(args):

    res = {
        "main_memory": 1000,
        "main_time_per_index": 1,
    # 'pass':{'constraint': 'knl', 'account': 'des', 'qos': 'regular'}} # Cori
    }

    return res

def setup(args):
    
    description = "Generate the indices of the four DES footprints"
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )

    parser.add_argument(
        "--dir_in", type=str, default="/global/cscratch1/sd/athomsen/des_y3_tests/000_test/DES_Y3KP_NGSF", help="input dir containing the catalog"
    )

    args = parser.parse_args(args)

    utils_logging.set_all_loggers_level(args.verbosity)

    return args

# no index dependence
def main(indices, args):
    # only needs to be ran once
    indices = 0

    args = setup(args)

    """ data loading """
    LOGGER.info(f"loading data")
    dir_resources = os.path.join(os.path.dirname(__file__), '..', 'resources')

    pix_list = []
    vec_list = []
    # tomo 0 is a sum of all bins, 1-4 the individual bins
    for tomo in range(n_z_bins):
        # J2000 angles in degrees
        alpha = h5py.File(os.path.join(args.dir_in, f'ALPHAWIN_J2000_tomo={tomo}.h5'))['ALPHAWIN_J2000'][:]
        delta = h5py.File(os.path.join(args.dir_in, f'DELTAWIN_J2000_tomo={tomo}.h5'))['DELTAWIN_J2000'][:]

        # conversion to healpy radian
        theta = -np.deg2rad(delta) + np.pi/2
        phi = np.deg2rad(alpha)

        # derived pixel ids, shape (num_galaxies,)
        pix = hp.ang2pix(nside=nside, theta=theta, phi=phi)

        # vector positions of the galaxies, shape (num_galaxies, 3)
        vec = hp.ang2vec(theta=theta, phi=phi)

        pix_list.append(pix)
        vec_list.append(vec)

    """ rotation of the original footprint """
    LOGGER.info(f"rotating the orignal footprint")
    # angles found by trial and error
    y_rot = get_rot_y(-0.125)
    z_rot = get_rot_z(-1.22)

    tomo_rotated_pix_ring = []
    # loop over tomographic bins
    for tomo, vec in enumerate(vec_list):
        # shape (n_galaxies, 3)
        rotated_vec = np.dot(y_rot, vec.T)

        # shape (3, n_galaxies)
        rotated_vec = np.dot(z_rot, rotated_vec)

        # convert the vectors to pixels
        rotated_pix_ring = hp.vec2pix(nside, rotated_vec[0], rotated_vec[1], rotated_vec[2])

        tomo_rotated_pix_ring.append(rotated_pix_ring)

    # remove double counting in the maps
    tomo_footprint_indices_ring = []
    for rotated_pix_ring in tomo_rotated_pix_ring:
        tomo_footprint_indices_ring.append(np.unique(rotated_pix_ring))

    """ HEALPix symmetry transformations to generate patches """
    from cosmogrid_des_y3.utils_healpix import rotate_pix, mirror_pix
    LOGGER.info(f"applying symmetry transforms")
    patches = []
    for i, patch_0 in enumerate(tomo_footprint_indices_ring):
        # perform symmetry transformations
        patch_1 = rotate_pix(patch_0, n_rot=2)
        patch_2 = mirror_pix(patch_0, lr=False)
        patch_3 = mirror_pix(patch_0, lr=False)
        patch_3 = rotate_pix(patch_3, n_rot=2)

        # all patches have the same size
        assert len(patch_0) == len(patch_1) == len(patch_2) == len(patch_3)
        # no index occurs more than once
        assert reduce(np.intersect1d, ([patch_0, patch_1, patch_2, patch_3])).size == 0

        patches.append(np.stack([patch_0, patch_1, patch_2, patch_3]))

    """ save the patches """
    LOGGER.info(f"saving the indices")
    with h5py.File(os.path.join(dir_resources, f"DES_Y3_patches_{nside}.h5"), "a") as f:
        f.attrs["info"] = f"This file contains the indices for nside {nside} in RING ordering of the four DES Y3 patches " \
        "(related through a HEALpix symmetry transformation) used to cut out the survey footprint four times from the full sky maps. " \
        "tomo 0 contains the combined elements in tomo bins 1-4"
        
        for i_bin in range(n_z_bins):
            current_patch = patches[i_bin]
            dset = f.create_dataset(f"tomo_{i_bin}", shape=current_patch.shape,
                                    dtype="i", data=current_patch)
            dset.attrs["info"] = "This dataset contains all pixel ids for a map with nside " \
                                f"{nside} in RING ordering, to cut out 4 DES Y3 data patches simulataneously. " \
                                "The shape is (N_patches, N_pix), the first patch is the original one. " \
                                f"Data is for tomo bin: {i_bin}"

    # only for esub-epipe compatibility
    yield

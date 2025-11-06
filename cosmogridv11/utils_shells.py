
# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2022
author: Tomasz Kacprzak
"""

import os, sys, warnings, argparse, h5py, numpy as np, time, logging, itertools, shutil
from cosmogridv11 import utils_io, utils_logging, utils_config, utils_cosmogrid, utils_arrays
from cosmogridv11.filenames import *
import healpy as hp

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)


def add_flips_and_rots(shells_group, perms_info):
    """
    Apply flips and rotations to the shell group
    """

    # apply permutations
    from cosmogridv11.utils_healpix import rotate_pix, mirror_pix
    healpix_pixel_indices = np.arange(shells_group.shape[1])

    if perms_info['rot'] > 0:
        healpix_pixel_mod = rotate_pix(pix=healpix_pixel_indices, 
                                       nside=hp.npix2nside(len(healpix_pixel_indices)), 
                                       n_rot=perms_info['rot'])

        LOGGER.info(f"applying rotation n_rot={perms_info['rot']}")
        shells_group = shells_group[:,healpix_pixel_mod]   

    if perms_info['flip_lr'] == 1:
        healpix_pixel_mod = mirror_pix(pix=healpix_pixel_indices, 
                                       nside=hp.npix2nside(len(healpix_pixel_indices)), 
                                       lr=True)

        LOGGER.info(f"applying flip_lr")
        shells_group = shells_group[:,healpix_pixel_mod]   

    if perms_info['flip_ud'] == 1:
        healpix_pixel_mod = mirror_pix(pix=healpix_pixel_indices, 
                                       nside=hp.npix2nside(len(healpix_pixel_indices)), 
                                       lr=False)

        LOGGER.info(f"applying flip_ud")
        shells_group = shells_group[:,healpix_pixel_mod]   

    return shells_group

def store_permutation_index(filepath_perm_index, perms_info, shell_groups, mode='w'):

    with h5py.File(filepath_perm_index, mode) as f:
        f.create_dataset(name='perms_info', data=perms_info)
        for i, shell_group in enumerate(shell_groups):
            f.create_dataset(name=f'shell_groups/{i}', data=shell_group)
    LOGGER.info(f'wrote {filepath_perm_index}')

# def load_permutation_index(filepath_perm_index):

#     shell_groups = []
#     with h5py.File(filepath_perm_index, 'r') as f:
#         perms_info = np.array(f['perms_info'])
#         for i in f['shell_groups'].keys():
#             shell_group = np.array(f[f'shell_groups/{i}'])        
#             shell_groups.append(shell_group)
#     return perms_info, shell_groups

def get_shell_permutation_sequence(n_shell_groups, n_sims):

    rots = [0, 1, 2, 3]
    flips_ud = [False, True]
    flips_lr = [False, True]

    perms_info = utils_arrays.zeros_rec(n_shell_groups, columns=['id_sim:i4', 'rot:i4', 'flip_ud:i4', 'flip_lr:i4'])
    perms_info['id_sim'] = perm_sequence(range(n_sims), n_shell_groups)
    mod_perms = perm_sequence(list(itertools.product(rots, flips_ud, flips_lr)), n_shell_groups)
    perms_info['rot'] = mod_perms[:,0]
    perms_info['flip_ud'] = mod_perms[:,1]
    perms_info['flip_lr'] = mod_perms[:,2]

    return perms_info



def perm_sequence(elements, length):

    perms = [np.random.permutation(elements) for _ in range(int(np.ceil(length/len(elements))))]    
    return np.concatenate(perms)[:length]

def get_shell_groups(shell_info, n_max_replicas, Lbox):

    # output container
    shell_groups = []

    # get shells groupings - up to crossing the replica box boundary
    for ir in range(0, n_max_replicas):

        # select shells which lie in side the current replica box
        replica_lower_com = ir * Lbox 
        replica_upper_com = (ir+1) * Lbox 

        LOGGER.info(f'replica: {ir+1}/{n_max_replicas: 3d}: {replica_lower_com: 6.2f} Mpc/h -> {replica_upper_com: 6.2f}Mpc/h')

        select1 = (shell_info['upper_com'] <= replica_upper_com) # upper edge of shell inside the box
        select2 = (shell_info['upper_com'] > replica_upper_com) & (shell_info['shell_com'] < replica_upper_com)  # upper edge outside, but center inside 

        select3 = (shell_info['lower_com'] > replica_upper_com) # upper edge of shell inside the box
        select4 = (shell_info['lower_com'] <= replica_upper_com) & (shell_info['shell_com'] > replica_lower_com) # upper edge outside, but center inside 

        select = (select1 | select2) & (select3 | select4)

        if np.count_nonzero(select) == 0:

            LOGGER.info('---> ran out of shells')
            continue

        shell_group = shell_info[select]
        shell_groups.append(shell_group)

        LOGGER.info(f"replica: {ir+1}/{n_max_replicas: 3d}  n_shells={len(shell_group): 4d}  id_lower={shell_group['shell_id'][0]: 4d} id_upper={shell_group['shell_id'][-1]: 4d} z_lower={shell_group['lower_z'][0]: 2.2f} z_upper={shell_group['upper_z'][-1]: 2.2f}")

    return shell_groups


def check_perms_completed(filepath_perm_index):

    if not os.path.isfile(filepath_perm_index):
        return False

    try:
        perms = load_permutation_index(filepath_perm_index)
        if len(perms[0]) == len(perms[1]):
            return True
        else:
            return False

    except Exception as err:
        LOGGER.error(f'failed to load {filepath_perm_index} errmsg={str(err)}')
        return False

    return False





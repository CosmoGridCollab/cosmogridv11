# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2022
author: Tomasz Kacprzak, Arne Thomsen
"""

import os, sys, warnings, argparse, h5py, numpy as np, time, logging, itertools, shutil
from cosmogrid_des_y3 import utils_io, utils_logging, utils_config
from cosmogrid_des_y3.filenames import *
import healpy as hp

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)
from UFalcon import probe_weights

# constants
TCMB0 = 2.7255  # cmb temperature
NEFF = 3.046  # effective number of neutrino species
from astropy.cosmology import FlatwCDM
from astropy.units import eV


def setup(args):

    description = "Cut out DES footprints"
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
        "--config", type=str, required=True, help="configuration yaml file"
    )
    parser.add_argument(
        "--dir_out", type=str, required=True, help="output dir for the results"
    )
    parser.add_argument("--test", action="store_true", help="test mode")
    args = parser.parse_args(args)

    utils_logging.set_all_loggers_level(args.verbosity)

    # get absolute paths
    args.config = utils_io.get_abs_path(args.config)
    args.dir_out = utils_io.get_abs_path(args.dir_out)

    return args


def resources(args):

    res = {
        "main_memory": 4000,
        "main_time_per_index": 4,  # hours
        "main_scratch": 6500,
        "merge_memory": 64000,
        "merge_time": 24,
    }
    # 'pass':{'constraint': 'knl', 'account': 'des', 'qos': 'regular'}} # Cori

    return res


def get_input_for_variant(variant):

    if variant == "baryonified":
        filename_shells = "baryonified_shells.npz"

    elif variant == "nobaryons":
        filename_shells = "shells_nside=512.npz"

    else:
        raise Exception(f"unknown analysis variant {variant}")

    return filename_shells


def store_probe_patches(filepath_out, probe_patches):

    with h5py.File(filepath_out, "w") as f:

        for probe in probe_patches.keys():
            for sample in probe_patches[probe].keys():
                for patch in probe_patches[probe][sample].keys():
                    dset = f"{probe}/{sample}/{patch}"
                    f.create_dataset(
                        name=dset,
                        data=probe_patches[probe][sample][patch],
                        compression="lzf",
                        shuffle=True,
                    )
                    LOGGER.info(f"stored {dset} in {filepath_out}")


def main(indices, args):
    """
    Cut out the four DES Y3 footprints from the full sky maps.
    Code lifted from Janis Fluri's Kids1000 analysis.
    https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/carpet/notebooks/KiDS_1000_patches.ipynb

    The DES Y3 specific patch indices are generated in
    https://cosmo-gitlab.phys.ethz.ch/arne.thomsen/cosmogrid_desy3/-/blob/main/notebooks/index_file.ipynb

    """

    args = setup(args)
    conf = utils_config.load_config(args.config)

    # get simulation list
    dir_resources = os.path.join(os.path.dirname(__file__), "..", "resources")
    simslist_grid = np.load(os.path.join(dir_resources, "CosmoGrid_grid_simslist.npy"))
    simslist_fiducial = np.load(os.path.join(dir_resources, "CosmoGrid_fiducial_simslist.npy"))
    parslist_grid = np.load(os.path.join(dir_resources, "CosmoGrid_grid_parslist.npy"))
    parslist_fiducial = np.load(os.path.join(dir_resources, "CosmoGrid_fiducial_parslist.npy"))
    LOGGER.info(f"got n_pars_grid={len(parslist_grid)} n_pars_fiducial={len(parslist_fiducial)} n_total={len(parslist_grid)+len(parslist_fiducial)}")
    LOGGER.info(f"got n_sims_grid={len(simslist_grid)} n_sims_fiducial={len(simslist_fiducial)} n_total={len(simslist_grid)+len(simslist_fiducial)}")

    # load the per redshift bin footprint patches structured like tomo_{0-4} where 0 includes 1-4
    with h5py.File(os.path.join(dir_resources, "DES_Y3_patches_512.h5"), "r") as tomo_patches_indices:

        # loop over sims
        for index in indices:

            sim_params = simslist_grid[index]

            # baryons, nobary
            for variant in conf["analysis_variants"]:

                LOGGER.info(f"==============> maps for variant={variant}")

                # output container
                probe_patches = {}

                # load the maps structured like {kg, ia, dg}/sourcegals_{1-4}
                maps_parent_path = get_dirname_projected_maps(args.dir_out, sim_params)
                maps_file_path = get_filepath_projected_maps(maps_parent_path, variant)

                with h5py.File(maps_file_path, "r") as maps:
                    # kg, ia, dg
                    for probe in maps.keys():
                        probe_patches[probe] = {}

                        n_z_bins = len(maps[probe].keys())
                        LOGGER.info(f"probe={probe} found {n_z_bins} redshift bins")

                        # loop over the 10 redshift bins (4 Metacalibration, 6 MagLim)
                        for sample in maps[probe].keys():
                            if "lensingsample" in sample:
                                # tomo_{1-4} excluding 0
                                patches_indices = tomo_patches_indices[f"tomo_{sample[-1]}"]

                            elif "maglimsample" in sample:
                                # combination of all tomographic bins
                                patches_indices = tomo_patches_indices["tomo_0"]

                            else:
                                raise Exception(f"unknown sample {sample}")

                            probe_patches[probe][sample] = {}

                            # convert to numpy for faster and fancy indexing
                            full_sky_map = maps[probe][sample][:]

                            # loop over patches
                            for i, patch_indices in enumerate(patches_indices):
                                probe_patches[probe][sample][f"patch_{i}"] = full_sky_map[patch_indices]

                # output files and store
                filepath_out = get_filepath_patches(maps_parent_path, variant)
                store_probe_patches(filepath_out, probe_patches)

            yield index

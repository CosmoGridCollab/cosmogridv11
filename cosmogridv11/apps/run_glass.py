"""Utility entry points for the GLASS pipeline.

This module mirrors the command line interface provided by
``run_probemaps`` but currently only implements the argument parsing and
basic application skeleton.
"""

from __future__ import annotations

import argparse, time
import os, sys
import numpy as np
from typing import Iterable, Sequence

from cosmogridv11 import utils_io, utils_logging, utils_config, utils_cosmogrid, utils_shells, utils_maps, utils_projection
from cosmogridv11.filenames import *

LOGGER = utils_logging.get_logger(__file__)



def setup(args: Sequence[str]) -> argparse.Namespace:
    """Parse command line arguments for the GLASS pipeline.

    Parameters
    ----------
    args:
        Sequence of command line arguments excluding the program name.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with absolute paths resolved where applicable.
    """

    description = "Run GLASS pipeline tasks"
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
        "--config",
        type=str,
        required=True,
        help="Configuration yaml file",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        required=False,
        default=None,
        help=(
            "Output directory for the results; use None for the current "
            "directory."
        ),
    )
    parser.add_argument(
        "--num_maps_per_index",
        type=int,
        default=20,
        help="Number of permutations per index to process",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable test mode",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip if output file exists (useful for crashed jobs)",
    )
    parser.add_argument(
        "--dir_out_archive",
        type=str,
        default=None,
        help="Optional archive directory where results will be copied",
    )
    parser.add_argument(
        "--largemem",
        action="store_true",
        help="Request additional memory resources",
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="Request extended runtime",
    )
    parser.add_argument(
        "--precopy",
        action="store_true",
        help="Copy all simulations before processing",
    )

    parsed = parser.parse_args(args)

    utils_logging.set_all_loggers_level(parsed.verbosity)

    parsed.config = utils_io.get_abs_path(parsed.config)
    if parsed.dir_out is not None:
        parsed.dir_out = utils_io.get_abs_path(parsed.dir_out)
    if parsed.dir_out_archive is not None:
        parsed.dir_out_archive = utils_io.get_abs_path(parsed.dir_out_archive)

    return parsed


def main(indices: Iterable[int], args: Sequence[str]) -> None:
    """Entry point for GLASS processing jobs.

    Parameters
    ----------
    indices:
        Iterable of integer job indices to be processed.
    args:
        Command line arguments provided to the application.
    """

    parsed_args = setup(args)
    LOGGER.info("Starting GLASS pipeline stub")
    LOGGER.debug("Indices received: %s", list(indices))

    config = utils_config.load_config(parsed_args.config)
    LOGGER.debug("Configuration keys: %s", list(config))

    workdir = os.environ.get("TMPDIR", os.getcwd())
    if parsed_args.dir_out is None:
        parsed_args.dir_out = workdir

    LOGGER.info("Working directory: %s", workdir)
    LOGGER.info("Output directory: %s", parsed_args.dir_out)


def main(indices, args):
    """
    Project shells using probe weights computed realier.
    Code lifted from Janis Fluri's Kids1000 analysis.
    https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/map_projection/patch_generation/project_patches.py#L1
    """

    args = setup(args)
    conf = utils_config.load_config(args.config)

    do_perms = conf['projection']['shell_perms']

    if do_perms:
        todolist_all = utils_cosmogrid.load_permutations_list(conf)
    else:
        todolist_all = utils_cosmogrid.get_simulations_list(set_type='all')[0]

    # change dir to temp
    tmp_dir = os.environ['TMPDIR'] if 'TMPDIR' in os.environ else os.getcwd()
    os.chdir(tmp_dir)
    LOGGER.info(f'changed dir to  {os.getcwd()}')
    if args.dir_out is None:
        args.dir_out = tmp_dir
    LOGGER.info(f'storing results in {args.dir_out}')

    
    # loop over sims 
    for index in indices: 

        LOGGER.info(f'==================================================> index={index} num_maps_per_index={args.num_maps_per_index}')
        time_start = time.time()

        todo_ids = np.arange(index*args.num_maps_per_index, (index+1)*args.num_maps_per_index)
        files_out_all = []

        if args.precopy:

            LOGGER.info('copying shells')
            filenames_all = []

            for variant in conf['analysis_variants']:

                # define simulation and local dirs
                filenames = list(np.unique(todolist_all['path_par'][todo_ids]))
                filenames = [f.split('bary')[1].lstrip('/') for f in filenames] # bary here is just a keyword for baryonified maps, which contains both dmb and dmo
                filename_shells, nside = get_filename_shells_for_variant('v11dmo') # it can also be v11dmb, does not matter
                filenames_all += [f'./{f}/*/{filename_shells}' for f in filenames]

            path_shells_local = utils_maps.copy_cosmogrid_file(conf, 
                                                               path_sim='', 
                                                               filename=filenames_all, 
                                                               check_existing=False, 
                                                               store_key='bary', # in this version, the "bary" file contains both dmo and dmb
                                                               rsync_args=' -v ',
                                                               rsync_R=True)

        for id_ in todo_ids:

            LOGGER.info(f'================================> projecting maps map_id={id_}')

            if do_perms:
                LOGGER.warning('no would project permuted maps, not implemented yet')
                raise NotImplementedError('not implemented yet')
            else:
                LOGGER.warning('no would project maps, not implemented yet')
                files_out = project_single_sim(index, args, conf)

            for f in files_out:
                utils_io.ensure_permissions(f, verb=True)

            files_out_all.extend(files_out)

        # if needed, copy to external archive and remove
        if args.dir_out_archive is not None:

            utils_io.archive_results(files_to_copy=files_out_all,
                                     dir_out=args.dir_out,
                                     dir_out_archive=args.dir_out_archive)


        LOGGER.info(f'cleaning up temp CosmoGrid files')
        utils_maps.cleanup_cosmogrid_files()
        LOGGER.info(f'done with index {index} time={(time.time()-time_start)/60.:2.2f} min')

        yield index


def project_single_sim(index, args, conf):
    """
    Project a single simulation using the probe weights computed realier.
    """

    simslist_all, parslist_all, shellinfo_all = utils_cosmogrid.get_baryonified_simulations_list(conf, set_type='all')
    sim_current = simslist_all[index]
    shellinfo_current = shellinfo_all[sim_current['path_par']]

    LOGGER.info(f"=============================> index={index} sim={sim_current['path_par']}")

    # get kernels
    nz_info = conf['redshifts_nz']

    # prepare output
    dirpath_out = get_dirname_projected_maps(args.dir_out, sim_current, id_run=sim_current['seed_index'], project_tag=conf['tag'])
    utils_io.robust_makedirs(dirpath_out)
    utils_io.ensure_permissions(dirpath_out, verb=True)

    files_variants = []
    for variant in conf['analysis_variants']:

        LOGGER.info(f'==============> maps for variant={variant}')
        
        # get the right input and load
        filename_shells, _ = get_filename_shells_for_variant(variant)
        filepath_shells_local = utils_maps.copy_cosmogrid_file(conf, path_sim=sim_current['path_sim'], filename=filename_shells, check_existing=args.test, store_key='bary')
        shells =  utils_maps.load_v11_shells(filepath_shells_local, variant)

        # main magic: project probes

        project_probes(shells['particles'], nz_info, sim_current, shellinfo_current)

        probe_cells = check_cls_for_probes(probe_maps, nz_info, sim_current)

        # output files and store
        filepath_out = get_filepath_projected_maps(dirpath_out, variant)
        LOGGER.info('storing maps')
        utils_maps.store_probe_maps(filepath_out, probe_maps, probe_cells=probe_cells, survey_mask=conf['projection']['survey_mask'], mode='w')
        LOGGER.info('storing kernels')
        utils_projection.store_probe_kernels(filepath_out, nz_info, w_shell, mode='a')

        files_variants.append(filepath_out)

    return files_variants


def get_filename_shells_for_variant(variant):

    # for CosmoGridV1.1
    if variant in ['v11dmb', 'v11dmo']:
        filename_shells = 'baryonified_shells_v11.h5'
        nside = 1024

    else:
        raise Exception(f'unknown analysis variant {variant}')

    return filename_shells, nside


def project_probes(shells, nz_info, sim_params, shellinfo, num_samples_per_shell=100):
    """
    Project probes using GLASS.
    """

    from cosmology.compat.camb import Cosmology
    import camb, glass

    h = sim_params['H0'] / 100
    camb_pars = camb.set_params(
        H0= sim_params['H0'],
        omch2=sim_params['O_cdm'] * h**2,
        ombh2=sim_params['Ob'] * h**2,
        omnuh2=sim_params['O_nu'] * h**2,
        NonLinear=camb.model.NonLinear_both,
        num_massive_neutrinos=3,
        mnu=sim_params['m_nu'],
        standard_neutrino_neff=True,
        w=sim_params['w0'],
        wa=sim_params['wa'],
        dark_energy_model='ppf',
        As=sim_params['As'],
        ns=sim_params['ns'],
    )
    cosmo = Cosmology(camb.get_background(camb_pars))

    for nz_ in nz_info:

        za = np.linspace(nz_['z'][0], nz_['z'][1], num_samples_per_shell)
        wa = np.ones_like(za)
        zeff = 0.5 * (nz_['z'][0] + nz_['z'][1])
        window = glass.RadialWindow(za=za, wa=wa, zeff=zeff)

        for probe in nz_['probes']:

            if probe == 'kg':

                shells_gamma = np.zeros((len(shells), len(shells[0])), dtype=np.complex128)
                convergence = glass.lensing.MultiPlaneConvergence(cosmo)

                for i in LOGGER.progressbar(range(len(shells)), desc='converting to delta to gamma'):

                    shell_delta = shells[i]/np.mean(shells[i]) - 1 # delta is the density contrast
                    
                    z_mid = (shellinfo[i]['lower_z'] + shellinfo[i]['upper_z']) / 2
                    convergence.add_window(shell_delta, window)
                    # get shear field
                    kappa = copy.deepcopy(convergence.kappa)
                    gamma = glass.lensing.from_convergence(kappa, lmax=nside_maps*3-1, shear=True)
                    shells_gamma[i] = gamma
                        
                import pudb; pudb.set_trace();



    raise NotImplementedError('not implemented yet')

    
    


    


if __name__ == '__main__':

    next(main([0], sys.argv[1:]))
# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2022
author: Tomasz Kacprzak
"""

import os, sys, warnings, argparse, h5py, numpy as np, time, logging, itertools, shutil
from cosmogridv1 import utils_io, utils_logging, utils_config, utils_cosmogrid, utils_arrays, utils_shell_permutation, utils_maps
from cosmogridv1.filenames import *
import healpy as hp
from UFalcon import probe_weights
from cosmogridv1.copy_guardian import NoFileException


warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

def setup(args):

    description = 'Make maps'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
                        help='logging level')
    parser.add_argument('--config', type=str, required=True, 
                        help='configuration yaml file')
    parser.add_argument('--dir_out', type=str, required=True, 
                        help='output dir for the results')
    parser.add_argument('--par_ids', type=str, required=True,
                        help='ids of parameters to run on')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    parser.add_argument('--resume', action='store_true',
                        help='skip if file exist, use if jobs crashed')
    parser.add_argument('--dir_out_archive', type=str, default=None, 
                        help='output dir for archiving the results, if specified the data will be copied there and removed from dir_out')

    args = parser.parse_args(args)

    utils_logging.set_all_loggers_level(args.verbosity)

    # get absolute paths
    args.config = utils_io.get_abs_path(args.config)
    args.dir_out = utils_io.get_abs_path(args.dir_out)


    return args


def resources(args):
    
    res = {'main_nsimult': 2000,
           'main_memory':24000,
           'main_time_per_index':4, # hours
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

    from cosmogridv1.copy_guardian import NoFileException

    args = setup(args)
    conf = utils_config.load_config(args.config)

    
    # loop over sims 

    for index in indices: 

        if conf['projection']['shell_perms'] == False:

            project_single_sim(index, args, conf)

        else:
            
            project_permuted_sims(index, args, conf)

        yield index

def get_input_for_variant(variant):

    if variant == 'baryonified512':
        filename_shells = 'baryonified_shells.npz'
        nside = 512
        
    elif variant == 'nobaryons512':
        filename_shells = 'shells_nside=512.npz'
        nside = 512

    elif variant == 'nobaryons2048':
        filename_shells = 'compressed_shells.npz'
        nside = 2048

    elif variant in ['v11dmb', 'v11dmo']:
        filename_shells = 'baryonified_shells_v11.h5'
        nside = 1024

    else:
        raise Exception(f'unknown analysis variant {variant}')

    return filename_shells, nside

def load_shells_for_variant(conf, path_sim, filename_shells, variant, check_existing):

    filepath_shells = os.path.join(conf['paths']['cosmogrid_root'], path_sim, filename_shells)
    filepath_shells_local = utils_maps.copy_cosmogrid_file(conf, path_sim, filename_shells, check_existing=check_existing)

    if variant in ['baryonified512', 'nobaryons512', 'nobaryons2048']:

        shells = utils_maps.load_compressed_shells(filepath_shells_local)

    elif variant in ['v11dmo', 'v11dmb']:

        shells =  utils_maps.load_v11_shells(filepath_shells_local, variant)

    else:
        
        raise Exception(f'unknown analysis variant {variant}')

    LOGGER.info(f'read shells with size {shells.shape} from {filepath_shells_local}')

    return shells








    # # check if there is anything to do
    # filepath_out = get_filepath_projected_maps(dirpath_out, variant)
    # if check_exists:
    #     maps_file_ready = utils_maps.check_maps_completed(conf, filepath_out, nside)
    #     if maps_file_ready:
    #         LOGGER.critical(f'maps exist and checked {filepath_out}')
    #         continue

    # try:
    #     # special_path_sim
    #     shells = utils_maps.load_shells(conf=conf, path_sim=sim_params['path_sim'], filename_shells=filename_shells)
    # except NoFileException as err:
    #     LOGGER.error(f'failed to load shells, err={err}')
    #     LOGGER.error(f'---> errors for index={index} variant={variant}, skipping...')
    #     continue

def arr_row_str(a):

    s = ''
    for k in a.dtype.names:
        s += f'{k}={str(a[k]):>4s} '
    return s

def project_single_sim(index, args, conf):


    simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_simulations_list(set_type='all')
    par_ids = utils_cosmogrid.parse_par_ids(args.par_ids, n_max=len(parslist_all))
    parslist_use = parslist_all[par_ids]
    index_par = index%len(parslist_use)
    index_perm = index//len(parslist_use)
    params_current = parslist_use[index_par]

    LOGGER.info(f"====================================> index={index} index_par={index_par} index_perm={index_perm} n_sims_use={n_sims_use} path_par={params_current['path_par']}")

    # prepare output
    dirpath_out = get_dirname_projected_maps(args.dir_out, params_current, id_run=index_perm)
    filename_weight = get_filename_probe_weights(dirpath_out)
    w_shell, probe_weights = utils_maps.load_probe_weigths(filename_weight, index_perm)
    probes = list(probe_weights.keys())
    LOGGER.info(f'using probes {probes}')
    nz_info = conf['redshifts_nz']


    for variant in conf['analysis_variants']:

        LOGGER.info(f'==============> maps for variant={variant}')
        
        # get the right input and load
        filename_shells, nside = get_input_for_variant(variant)

        # check if there is anything to do
        filepath_out = get_filepath_projected_maps(dirpath_out, variant)
        if args.resume:
            maps_file_ready = utils_maps.check_maps_completed(conf, filepath_out, nside)
            if maps_file_ready:
                LOGGER.critical(f'maps exist and checked {filepath_out}')
                continue

        try:
            
            # special_path_sim
            shells = load_shells_for_variant(conf, path_sim=sim_params['path_sim'], filename_shells=filename_shells, variant=variant, check_existing=args.test)
            # shells = utils_maps.load_shells(conf=conf, path_sim=sim_params['path_sim'], filename_shells=filename_shells)
        
        except NoFileException as err:
            
            LOGGER.error(f'failed to load shells, err={err}')
            LOGGER.error(f'---> errors for index={index} variant={variant}, skipping...')
            
            continue

        probe_maps = utils_maps.project_all_probes(shells, probes, probe_weights, 
                                                   nside=nside, 
                                                   n_particles=sim_params['box_size_Mpc_over_h'], 
                                                   box_size=sim_params['n_particles'])


        # add high redshift shell using Gaussian Random Field from stored cls
        probe_maps = utils_maps.add_highest_redshift_shell(probe_maps, nz_info, sim_params=sim_params, parslist_all=parslist_all)

        del(shells)

        # output files and store
        utils_maps.store_probe_maps(filepath_out, probe_maps)


def project_permuted_sims(index, args, conf):
    """
    :param index: runs over cosmological parameters sets
    """

    from cosmogridv1 import baryonification

    # cosmogrid specs
    n_max_replicas = int(conf['projection']['n_max_replicas'])
    n_max_shells = int(conf['projection']['n_max_shells'])

    simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_simulations_list(set_type='all')
    par_ids = utils_cosmogrid.parse_par_ids(args.par_ids, n_max=len(parslist_all))
    parslist_use = parslist_all[par_ids]
    index_par = index%len(parslist_use)
    index_perm = index//len(parslist_use)

    # index calculations
    params_current = parslist_use[index_par] # this doesn't use anything about the sim, just the params, this is here to match the indexing scheme with the "no permutation" version
    path_params_current = params_current['path_par']
    shell_info_current = shell_info_all[path_params_current]
    n_sims_avail = np.count_nonzero(simslist_all['path_par']==path_params_current)
    n_sims_use = min(float(conf['projection']['n_max_sims_use']), n_sims_avail)
        
    # get permutation indices range to run on
    LOGGER.info(f"====================================> index={index} index_par={index_par} index_perm={index_perm} n_sims_use={n_sims_use} path_par={params_current['path_par']}")

    # define simulation and local dirs
    tmp_dir = os.environ['TMPDIR'] if 'TMPDIR' in os.environ else os.getcwd()
    os.chdir(tmp_dir)
    LOGGER.info(f'changed dir to  {os.getcwd()}')
    
    # divide the shells into groups that span a single box
    shell_groups = utils_shell_permutation.get_shell_groups(shell_info_current, n_max_replicas, Lbox=params_current['box_size_Mpc_over_h'])
    n_shell_groups = len(shell_groups)

    # calculate permutations list
    seed = conf['projection']['shell_perms_seed']+index_perm
    np.random.seed(seed)
    LOGGER.info(f'using random seed={seed} for shell permutations')
    list_perms_info = []

    # output 
    dirname_out = get_dirname_permuted_maps(dir_out=args.dir_out,
                                            project_tag=conf['tag'],
                                            cosmo_params=params_current,
                                            id_perm=index_perm)
    utils_io.robust_makedirs(dirname_out)
    filepath_perm_index = get_filepath_permutations_index(dirname_out)

    # check if exists
    perms_file_ready = False
    if args.resume:
    
        LOGGER.info(f'resuming mode, file exists {filepath_perm_index}')
        perms_file_ready = utils_shell_permutation.check_perms_completed(filepath_perm_index)

    # calculate and store the permutation sequence
    if not perms_file_ready:
    
        perms_info = utils_shell_permutation.get_shell_permutation_sequence(n_shell_groups, n_sims_use)

        if args.test:
            LOGGER.warning('>>>>>>>>>>>>>>>> TEST mode, switching off permutations and shell shuffling!')
            perms_info[:] = 0, 0, False, False

        list_perms_info.append(perms_info)
        utils_shell_permutation.store_permutation_index(filepath_perm_index, perms_info, shell_groups)

    # main magic - get the projected maps from permuted sims
    project_single_permuted_sim(conf, args, filepath_perm_index, params_current, id_perm=index_perm, parslist_all=parslist_all)

    # if needed, copy to external archive and remove
    if args.dir_out_archive is not None:

        utils_io.archive_results(files_to_copy=[filepath_perm_index],
                                 dir_out=args.dir_out,
                                 dir_out_archive=args.dir_out_archive)


def project_single_permuted_sim(conf, args, filepath_perm_index, sim_params, id_perm, parslist_all):
        
    # report
    LOGGER.info(f"=================>  path_par={sim_params['path_par']}")

    # load the shell groups and permutations
    perms_info, shell_groups = utils_shell_permutation.load_permutation_index(filepath_perm_index)
    
    # get probe weights   

    # get output dir
    dirpath_out = get_dirname_permuted_maps(args.dir_out, 
                                            cosmo_params=sim_params, 
                                            project_tag=conf['tag'], 
                                            id_perm=id_perm)

    filename_weight = get_filename_probe_weights(dirpath_out)
    shell_weights, probe_weights = utils_maps.load_probe_weigths(filename_weight, conf)
    nz_info = conf['redshifts_nz']

    for variant in conf['analysis_variants']:

        LOGGER.info(f"============> creating permuted lightcone maps for variant={variant}, permutation id_sims={perms_info['id_sim']}")

        # get the right input and load
        filename_shells, nside = get_input_for_variant(variant)

        # check if exists
        filepath_out = get_filepath_projected_maps(dirpath_out, variant)
        if args.resume:
            maps_file_ready = utils_maps.check_maps_completed(conf, filepath_out, nside)
            if maps_file_ready:
                LOGGER.critical(f'maps exist {filepath_out}')
                continue



        # loadd all the shells from different simulations according to the shell group permutation index
        shells_perm = []
        success = True
        for i, id_sim in enumerate(perms_info['id_sim']):

            # load shells from the right file and select the shell needed
            LOGGER.info(f"======> loading sim {i+1}/{len(perms_info)}, id_sim={perms_info['id_sim'][i]}")
            path_sim = utils_cosmogrid.get_sims_path(sim_params, id_sim=perms_info['id_sim'][i])

            try:
                shells = load_shells_for_variant(conf, path_sim=path_sim, filename_shells=filename_shells, variant=variant, check_existing=args.test)
            except NoFileException as err:
                LOGGER.error(f'failed to load shells, err={err}')
                LOGGER.error(f'---> errors for id_perm={id_perm} variant={variant}, skipping...')
                success = False
                break

            # select the relevant shell group according to perm indices
            select_group = shell_groups[i]['shell_id']
            LOGGER.info('select_group='+str(select_group))
            shells_group = shells[select_group,:]

            # random flips and rotations on the shell group
            shells_group = utils_shell_permutation.add_flips_and_rots(shells_group, perms_info[i])

            # store
            shells_perm.append( shells_group )
            del(shells)

        if not success:
            LOGGER.warning(f'unsuccessful shell group creation for variant {variant}, skipping..')
            continue

        # stack all shels to create the permuted lightcone
        shells = np.concatenate(shells_perm, axis=0)
        LOGGER.info(f'stacked shell groups size={shells.shape}')

        probe_maps = utils_maps.project_all_probes(shells, nz_info, probe_weights, shell_weights, 
                                                   nside=nside, 
                                                   n_particles=sim_params['n_particles'], 
                                                   box_size=sim_params['box_size_Mpc_over_h'])


        # add high redshift shell using Gaussian Random Field from stored cls
        probe_maps = utils_maps.add_highest_redshift_shell(probe_maps, nz_info, sim_params=sim_params, parslist_all=parslist_all)

        # output files and store
        utils_maps.store_probe_maps(filepath_out, probe_maps)

def get_shell_distances_old(path_logs):
    """
    Uses extracted files 'CosmoML.log' and 'baryonification_params.py'
    :return shell_info_cov: shell info rec array with fields 'id', 'z_min', 'z_max', 'shell_cov', 'cov_inner', 'cov_outer'
    """

    from cosmogridv1 import baryonification
    from cosmogridv1.baryonification import halocone


    # get z bounds for shells
    log_file = os.path.join(path_logs, "CosmoML.log")  # log file of the simulation
    z_bounds = halocone.extract_redshift_bounds(log_file)
    sorting = np.argsort(z_bounds[:,0])
    z_bounds = z_bounds[sorting,:]
    n_shells = len(z_bounds)
    LOGGER.info(f"Extracted redshift bounds for n_shells={n_shells}")
    LOGGER.debug("Extracted redshift bounds: " + str(z_bounds))

    # load sim parameters and build cosmology
    sys.path.insert(0,path_logs)
    from baryonification_params import par
    cosmo = baryonification.utils.build_cosmo(param=par)

    # calcualte distances to shell boundaries

    shell_info_cov = utils_arrays.zeros_rec(n_shells, columns=['shell_id:i4', 'lower_z', 'upper_z', 'shell_com', 'lower_com', 'upper_com'])
    for i in range(len(z_bounds)):
        shell_info_cov[i] = i, z_bounds[i][0], z_bounds[i][1], *baryonification.halo_utils.get_shell_cov_dist(z_bounds[i], cosmo)


    return shell_info_cov, par




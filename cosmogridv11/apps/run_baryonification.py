# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""

Created July 2022
author: Tomasz Kacprzak
"""

import io, os, sys, warnings, argparse, h5py, numpy as np, time, logging, itertools, shutil, yaml, tempfile
import healpy as hp
from glob import glob
from tqdm import tqdm, trange
from cosmogridv1 import utils_logging, utils_io, utils_arrays, utils_cosmogrid, utils_config, utils_maps
from cosmogridv1.copy_guardian import NoFileException
from cosmogridv1.filenames import *

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

def show_help():

    help_text = """
    run_baryonification usage instructions
    ======================================

    Objective: baryonify cosmogrid shells for all simulations

    The app is ran in two stages:
    (1) create halocone with profiled halos, use command: profile_halos
    (2) create baryonified maps, use command displace_shells
        
    Individual commands:

    (1) run_baryonification profile_halos
    Input: halos snapshots, metadata
    Output: halocone file with profiled halos

    (2) run_baryonification displace_shells
    Input: halocone from step (1), particle shells, metadata
    Output: shells with displaced particles

    """
    print(help_text)


def setup(args):

    description = 'Bayronify shells'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('command', type=str, default='help', choices=('help', 'profile_halos', 'displace_shells'), 
                        help='command to run, "help" for instructions')
    parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
                        help='logging level')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    parser.add_argument('--long', action='store_true',
                        help='if to run on the long queue, 7 days')
    parser.add_argument('--config', type=str, required=True, 
                        help='configuration yaml file')
    parser.add_argument('--dir_out', type=str, required=True, 
                        help='output dir for the results')
    parser.add_argument('--dir_out_archive', type=str, default=None, 
                        help='output dir for archiving the results, if specified the data will be copied there and removed from dir_out')

    args, _  = parser.parse_known_args(args)

    utils_logging.set_all_loggers_level(args.verbosity)

    args.config = utils_io.get_abs_path(args.config)
    args.dir_out = utils_io.get_abs_path(args.dir_out)
    utils_io.robust_makedirs(args.dir_out)

    return args


def resources(args):

    if type(args) is list:
        args = setup(args)

    if args.command == 'profile_halos':
        
        main_memory = 48000 if args.long else 24000
    
    elif args.command == 'displace_shells':

        main_memory = 32000
        
    res = {'main_memory': main_memory,
           'main_time_per_index': 24*5 if args.long else 24, # hours
           'main_scratch':6500,
           'merge_memory':64000,
           'merge_time':24}

    if args.command == 'displace_shells':

        res['main_nsimult'] = 2000
    
    return res

def update_params(params_bary, simsinfo):
    """This is to fix a bug in baryonification_params.py, where the Mc parameter has the wrong label nu.
    It is overwritten by default from params.py
    
    From this file:

    par.baryon.nu      = 66000000000000.0
    par.baryon.nu      = 0.0
    
    Parameters
    ----------
    params_bary : module
        loaded baryonification_params.par module
    simsinfo : np.array rec
        line from the simsinfo/parsinfo cosmogrid meta table
    """

    params_bary.baryon.Mc = simsinfo['bary_Mc']
    LOGGER.debug(f'fixed params_bary.baryon.Mc {params_bary.baryon.Mc:e}')

    return params_bary



# @profile
def main(indices, args):

    args = setup(args)
    conf = utils_config.load_config(args.config)

    simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_simulations_list(set_type='all')

    tmp_dir = os.environ['TMPDIR'] if 'TMPDIR' in os.environ else os.getcwd()
    os.chdir(tmp_dir)
    LOGGER.info(f'changed dir to  {os.getcwd()}')

    filename_baryparams = os.path.join(os.getcwd(), 'baryonification_params.py')
    
    if args.command == 'help':

        show_help()
        sys.exit(0)

    for index in indices:

        path_sim = simslist_all['path_sim'][index]
        path_par = simslist_all['path_par'][index]
        shell_info = shell_info_all[path_par]
        dir_out_current = os.path.join(args.dir_out, path_sim)
        utils_io.robust_makedirs(dir_out_current)
        
        LOGGER.debug('simulation params:')
        for c in simslist_all.dtype.names: 
            LOGGER.debug(f'{c:>20s} = {str(simslist_all[c][index]):<50s}')

        # prepare input
        filename_params_local = utils_maps.copy_cosmogrid_file(conf=conf, path_sim=path_sim, filename='param_files.tar.gz', check_existing=True)
        utils_maps.copy_cosmogrid_file(conf=conf, path_sim=path_sim, filename='params.yml', check_existing=True)
        decompress_params_files(filename_params_local)
        params_bary = load_baryonification_params(filename_baryparams)
        params_bary = update_params(params_bary, simsinfo=simslist_all[index])

        if args.command == 'profile_halos':

            filename_halocone_out =  get_filename_profiled_halos(dir_out=dir_out_current, tag=conf['baryonification']['tag'])

            # for delta simulations with deltas in baryon parameter, just copy the fiducial
            if 'delta_bary' in simslist_all['delta'][index]:
                
                copy_corresponding_fiducial_halocone(args, conf, index, simslist_all, filename_halocone_out)

            else:

                filename_halos_local = utils_maps.copy_cosmogrid_file(conf=conf, path_sim=path_sim, filename='pkd_halos.tar.gz', check_existing=True, noguard=args.test)
                decompress_halos_files(filename_halos_local)
        
                # main magic - get the halos
                halo_data, shell_data = gen_halocone(conf,  shell_info, params_bary, test=args.test)

                # store
                store_profiled_halos(filename_halocone_out, halo_data, shell_data, params_bary)


        elif args.command == 'displace_shells':

            if 'delta_bary' in simslist_all['delta'][index]:
                    
                # replace delta_bary_* params with fiducial
                path_sim = path_sim.replace(simslist_all['delta'][index], 'fiducial')
                LOGGER.info(f'this is a delta_bary simulation, using maps from cosmo_fiducial {path_sim}')

            filename_shells_local = utils_maps.copy_cosmogrid_file(conf=conf, path_sim=path_sim, filename="compressed_shells.npz", check_existing=True)
    
            # main magic - baryonify shells
            filename_halocone_out =  get_filename_profiled_halos(dir_out=dir_out_current, tag=conf['baryonification']['tag'])
            filename_shells_out =  get_filename_baryonified_shells(dir_out=dir_out_current, tag=conf['baryonification']['tag'])
            filename_bary_info_out =  get_filename_baryonification_info(dir_out=dir_out_current, tag=conf['baryonification']['tag'])

            baryon_shells, particle_shells, shell_stats_arr = baryonify_shells(conf, filename_halocone_out, filename_shells_local, params_bary, test=args.test)

            # write summary file
            create_baryonification_info_file(filename_bary_info_out, shell_stats_arr, params_bary)

            # store
            store_baryonified_shells(filename_shells_out, baryon_shells, particle_shells, shell_stats_arr)

            # if needed, copy to external archive and remove
            if args.dir_out_archive is not None:

                utios_io.archive_results(files_to_copy=[filename_shells_out],
                                         dir_out=args.dir_out,
                                         dir_out_archive=args.dir_out_archive)

        yield index




        # yield index


    # for index in indices:

    #     path_sim = simslist_all['path_sim'][index]
    #     path_par = simslist_all['path_par'][index]
    #     shell_info = shell_info_all[path_par]
    #     dir_out_current = os.path.join(args.dir_out, path_sim)
    #     utils_io.robust_makedirs(dir_out_current)

    #     try:
    #         filename_params_local = utils_maps.copy_cosmogrid_file(conf=conf, path_sim=path_sim, filename='param_files.tar.gz', check_existing=True)
    #         filename_halos_local = utils_maps.copy_cosmogrid_file(conf=conf, path_sim=path_sim, filename='pkd_halos.tar.gz', check_existing=True)
    #         filename_shells_local = utils_maps.copy_cosmogrid_file(conf=conf, path_sim=path_sim, filename=compressed_shells, check_existing=True)
    #         success = True
        
    #     except NoFileException as err:
    #         LOGGER.error(f'failed to load shells, err={err}')
    #         LOGGER.error(f'---> errors for index={index}, skipping...')
    #         success = False
        
    #     if success:

    #         try:

    #             # prepare input
    #             decompress_baryonification_files(filename_params_local, filename_halos_local)

    #             # get the halos
    #             filename_halos_out =  get_filename_profiled_halos(dir_out=dir_out_current, tag=conf['baryonification']['tag'])
    #             gen_halocone(conf,  shell_info, filename_halos_out, test=args.test)

    #             # main magic - baryonify shells
    #             filename_shells_out =  get_filename_baryonified_shells(dir_out=dir_out_current, tag=conf['baryonification']['tag'])
    #             baryonify_shells(conf, filename_halos_out, filename_shells_local, filename_shells_out, test=args.test)

    #         except NoFileException as err:
            
    #             LOGGER.error(f'Failed to load shells, err={err}')
    #             LOGGER.error(f'---> errors for index={index}, skipping...')
    #             continue


def copy_corresponding_fiducial_halocone(args, conf, index, simslist_all, filename_halocone_out):


    # get the file name
    index_fiducial = index % 200 # number of fiducial sims is 200
    path_sim_copy = simslist_all['path_sim'][index_fiducial]
    dir_copy = os.path.join(args.dir_out, path_sim_copy)
    filename_halocone_copy =  get_filename_profiled_halos(dir_out=dir_copy, tag=conf['baryonification']['tag'])
    LOGGER.info(f'simulation {index} is a delta for baryon parameters, copying fiducial halos: {filename_halocone_copy} -> {filename_halocone_out}')
        
    # check if exists
    if not os.path.isfile(filename_halocone_copy):
        raise Exception(f'missing fiducial halos file {filename_halocone_copy}')
    
    import shutil
    shutil.copyfile(filename_halocone_copy, filename_halocone_out)


def store_baryonified_shells(filename_out, baryon_shells, particle_shells, shell_infos_arr):

    nside_out = hp.npix2nside(baryon_shells.shape[1])
    nside_in = hp.npix2nside(particle_shells.shape[1])

    LOGGER.info(f'storing output {nside_in} -> {nside_out} particle_shells=[{particle_shells.shape}, {particle_shells.dtype}]  baryon_shells=[{baryon_shells.shape} {baryon_shells.dtype}]')

    # orig_shell = hp.ud_grade(particle_shells.astype(np.uint32), nside_out=nside_out, power=-2)
    orig_shell = particle_shells if nside_out == nside_in else np.array([hp.ud_grade(p.astype(np.uint32), nside_out=nside_out, power=-2) for p in particle_shells])
    LOGGER.info(f'orig shell {orig_shell.shape} {orig_shell.dtype}')

    # convert to diff
    for i in range(len(baryon_shells)):
        baryon_shells[i] -= orig_shell[i]
    diff_shell = baryon_shells
    LOGGER.info(f'diff shell {diff_shell.shape} {diff_shell.dtype}')

    LOGGER.info(f'storing {filename_out} {diff_shell.nbytes/1e9:4.2f} GB')

    compression_args = dict(compression='gzip', shuffle=True, compression_opts=4)

    diff_shell_inds = np.uint32(np.nonzero(diff_shell.ravel())[0])
    diff_shell_vals = diff_shell.ravel()[diff_shell_inds]
    time_start = time.time()
    with h5py.File(filename_out, 'w') as f:
        f.create_dataset(name='nobaryon_shells', data=orig_shell,      **compression_args)
        f.create_dataset(name='diff_shell_inds', data=diff_shell_inds, **compression_args)
        f.create_dataset(name='diff_shell_vals', data=diff_shell_vals, **compression_args)
        f.create_dataset(name='shell_dicts',     data=shell_infos_arr, **compression_args)

    LOGGER.info(f'stored {filename_out}, compression time {time.time()-time_start:4.0f} sec')


def store_profiled_halos(filename_out, halo_data, shell_data, param):

    # to save storage, only save [x, y, z, ids..] for all halos on the lightcone
    # the other parameters are stored onece per unique halo ID
    cols_lightcone = ['x', 'y', 'z', 'shell_id', 'halo_buffer']
    cols_base = list(set(halo_data.dtype.names)-set(cols_lightcone))
    uval, uind, uinv = np.unique(halo_data['ID'], return_index=True, return_inverse=True)
    halo_base = halo_data[uind] # select unique halos
    halo_base = utils_arrays.rewrite(halo_base[cols_base]) # use only base columns
    halo_pos = utils_arrays.rewrite(halo_data[cols_lightcone]) # use only lightcone columns
    halo_pos = utils_arrays.add_cols(halo_pos, names=['uid:i8'], data=[uinv]) # add unique id

    compression_args = dict(compression='lzf', shuffle=True)
    with h5py.File(filename_out, 'w') as f:
        f.create_dataset(name='halo_base', data=halo_base, **compression_args)
        f.create_dataset(name='halo_pos', data=halo_pos, **compression_args)
        f.create_dataset(name='shell_data', data=shell_data, **compression_args)
        f['halo_base'].attrs['MinParts'] = param.sim.Nmin_per_halo

    LOGGER.error(f'stored {filename_out}')

    halos_check = read_profiled_halos(filename_out)

    for c in halo_data.dtype.names:
        assert np.all(halo_data[c]==halos_check[c]), f'halo catalog compression failed for column {c}'

def read_profiled_halos(filename):

    cols_lightcone = ['x', 'y', 'z', 'shell_id', 'halo_buffer']

    # load the halos
    with h5py.File(filename) as f:
        shells = np.array(f["shell_data"])
        halo_pos = np.array(f["halo_pos"])
        halo_base = np.array(f["halo_base"])
        halos_MinParts = f["halo_base"].attrs['MinParts']

    # outer join
    halo_base = utils_arrays.add_cols(halo_base, names=cols_lightcone)
    halos = halo_base[halo_pos['uid']]
    for c in cols_lightcone:
        halos[c] = halo_pos[c]

    LOGGER.info(f'read {len(halos)} from {filename}')

    return halos



def create_baryonification_info_file(fname_out, shell_infos, param):

    # accumulate stats
    from cosmogridv1.baryonification.utils import bunch_to_lists

    # create the info file
    with open(fname_out, "w+") as f:
        f.write("Number of shells:       {} \n".format(len(shell_infos)))
        f.write("Number of unique halos: {} \n".format(np.sum(shell_infos['n_halos_unique'])))
        f.write("Total number of halos:  {} \n".format(np.sum(shell_infos['n_halos'])))
        f.write("Number of unique parts: {} \n".format(np.sum(shell_infos['n_parts_unique'])))
        f.write("Total number of parts:  {} \n".format(np.sum(shell_infos['n_parts'])))
        f.write("\n")
        f.write("Baryonification params: \n")
        f.write("======================= \n")
        lists = bunch_to_lists(param)
        for main_key, sub_group in lists:
            f.write("{}: \n".format(main_key))
            for sub_key, sub_val in sub_group:
                f.write("{}:".format(sub_key).ljust(20) + "{} \n".format(sub_val))
        f.write("\n")
        f.write("Individual shells: \n")
        f.write("================== \n")
        for shell_info in shell_infos:
            f.write("\n")
            f.write("Name of the shell:      {} \n".format(shell_info['shell_path']))
            f.write("Number of unique halos: {} \n".format(shell_info['n_halos_unique']))
            f.write("Total number of halos:  {} \n".format(shell_info['n_halos']))
            f.write("Number of unique parts: {} \n".format(shell_info['n_parts_unique']))
            f.write("Total number of parts:  {} \n".format(shell_info['n_parts']))

    LOGGER.info(f'wrote {fname_out}')

    return fname_out

def decompress_params_files(filename_params):

    import tarfile
    with tarfile.open(filename_params, "r") as tf:
        tf.extract(member='CosmoML.log')
        tf.extract(member='baryonification_params.py')
        tf.extract(member='class_processed.hdf5')

def decompress_halos_files(filename_halos_local):
    
    import tarfile
    with tarfile.open(filename_halos_local, "r") as tf:
        tf.extractall()


def load_baryonification_params(fname):

    from cosmogridv1 import baryonification
    sys.path.append(os.path.join(baryonification.__path__[0], '..')) # this is for back-compat with baryonification_params.py scripts
    import importlib.util
    spec = importlib.util.spec_from_file_location("baryonification_params.par", fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["baryonification_params.par"] = mod
    spec.loader.exec_module(mod)
    LOGGER.info(f'using baryonification_params={mod}')
    return mod.par


def gen_halocone(conf, shell_info, params_bary, test=False):
    """
    https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/carpet/baryons/gen_halocone.py
    This file creates a HaloCone from a CosmoGrid simulation.
    """


    from cosmogridv1 import baryonification

    # initialise parameters
    # sys.path.append('.') 
    # sys.path.append(os.path.join(baryonification.__path__[0], '..')) # this is for back-compat with baryonification_params.py scripts
    # par = load_baryonification_params(filename_params)
    # import baryonification_params
    
    # from baryonification_params import par

    # get redshift bounds
    z_bounds = np.stack([shell_info['lower_z'], shell_info['upper_z']], axis=1)
    LOGGER.info("Extracted redshift bounds:\n{}".format(z_bounds))

    # match the corresponding HaloFiles
    halo_files = baryonification.utils.get_halofiles_from_z_boundary(z_boundaries=z_bounds, 
                                                                     log_file="CosmoML.log", 
                                                                     prefix="./pkd_halos/CosmoML")
    LOGGER.info("Matched Halofiles:{}".format(halo_files))

    # write
    halo_data, shell_data = baryonification.prep_lightcone_halos(param=params_bary,
                                                                 halo_files=halo_files,
                                                                 shells_z=z_bounds,
                                                                 test=test)

    return halo_data, shell_data

    

def read_shells(filename_shells):

    with np.load(filename_shells) as f:

        LOGGER.info("Loading shells...")
        particle_shells = np.array(f['shells'])
        particle_shells_info = np.array(f['shell_info'])
        LOGGER.info("Done...")

    return particle_shells, particle_shells_info
    
# @profile
def baryonify_shells(conf, filename_halocone, filename_shells, params_bary, test=False):
    """
    https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/carpet/baryons/baryonify_shells.py
    This runs baryonification.
    """

    from cosmogridv1 import baryonification

    """
    This script baryonifies the compressed shells of a CosmoGrid simulation given a HaloCone and parameters
    """

    halos = read_profiled_halos(filename_halocone)

    particle_shells, particle_shells_info = read_shells(filename_shells)

    # create the halos
    baryon_shells, shell_stats_arr, shells_back = baryonification.get_baryon_shells(params_bary, 
                                                                                    shells=particle_shells,
                                                                                    shells_info=particle_shells_info,
                                                                                    halos=halos,
                                                                                    delta_shell=False,
                                                                                    nside_out=int(conf['baryonification']['nside_out']),
                                                                                    test=test,
                                                                                    interp_halo_displacements=conf['baryonification']['interp_halo_displacements'],
                                                                                    interp_ngrid=int(2**15))


    return baryon_shells, particle_shells, shell_stats_arr


def get_indices(tasks):
    """
    Parses the jobids from the tasks string.

    :param tasks: The task string, which will get parsed into the job indices
    :return: A list of the jobids that should be executed
    """
    # parsing a list of indices from the tasks argument

    if '>' in tasks:
        tasks = tasks.split('>')
        start = tasks[0].replace(' ', '')
        stop = tasks[1].replace(' ', '')
        indices = list(range(int(start), int(stop)))
    elif ',' in tasks:
        indices = tasks.split(',')
        indices = list(map(int, indices))
    else:
        try:
            indices = [int(tasks)]
        except ValueError:
            raise ValueError("Tasks argument is not in the correct format!")

    return indices

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, default='[0]')
    args, args_remaining = parser.parse_known_args(sys.argv[1:])

    next(main(indices=get_indices(args.tasks), args=args_remaining))



def missing(indices, args):

    args = setup(args)
    conf = utils_config.load_config(args.config)

    simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_simulations_list(set_type='all')

    tmp_dir = os.environ['TMPDIR'] if 'TMPDIR' in os.environ else os.getcwd()
    os.chdir(tmp_dir)
    LOGGER.info(f'changed dir to  {os.getcwd()}')

    filename_baryparams = os.path.join(os.getcwd(), 'baryonification_params.py')

    list_missing = []
    
    for index in indices:

        path_sim = simslist_all['path_sim'][index]
        dir_out_current = os.path.join(args.dir_out, path_sim)

        if args.command == 'profile_halos':

            filename_halocone_out =  get_filename_profiled_halos(dir_out=dir_out_current, tag=conf['baryonification']['tag'])

            if os.path.isfile(filename_halocone_out):

                LOGGER.debug(f'index={index:>8d} file OK {filename_halocone_out}')

            else:

                LOGGER.info(f'index={index:>8d} missing file {filename_halocone_out}')                
                list_missing.append(index)

        elif args.command == 'displace_shells':

            raise Exception('TODO missing for shell files')


    LOGGER.info(f'missing {len(list_missing)} indices: {str(list_missing)}')

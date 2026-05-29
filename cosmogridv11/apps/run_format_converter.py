# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created December 2025
author: Tomasz Kacprzak
"""

import os, warnings, argparse, h5py, numpy as np, time
from cosmogridv11 import utils_io, utils_logging, utils_config, utils_cosmogrid, utils_shells, utils_maps, utils_projection, utils_arrays
from cosmogridv11.filenames import *
import healpy as hp
from cosmogridv11.copy_guardian import NoFileException

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

STORE_COLS = ['As', 'H0', 'O_cdm', 'O_nu', 'Ob', 'Ol', 'Om', 'ns', 'w0', 'wa']
NSIDE_OUT = 256


def setup(args):

    description = 'Make maps'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
                        help='logging level')
    parser.add_argument('--config', type=str, required=True, 
                        help='Configuration yaml file')
    parser.add_argument('--dir_out', type=str, required=False, default=None, 
                        help='Output dir for the results, use None for current directory. It will be local job scratch if used on a cluster.')
    parser.add_argument('--num_maps_per_index', type=int, default=20,
                        help='Number of permutations per index to process')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    parser.add_argument('--resume', action='store_true',
                        help='Skip if file exist, use if jobs crashed')
    parser.add_argument('--dir_out_archive', type=str, default=None, 
                        help='Output dir for archiving the results, if specified the data will be copied there and removed from dir_out')
    parser.add_argument('--largemem', action='store_true',
                        help='Use more memory')
    parser.add_argument('--long', action='store_true',
                        help='Use more time')

    args = parser.parse_args(args)

    utils_logging.set_all_loggers_level(args.verbosity)

    if args.dir_out is not None:
        args.dir_out = utils_io.get_abs_path(args.dir_out)


    return args


def resources(args):

    if type(args) is list:
        args = setup(args)
    
    res = {'main_nsimult': 10,
           'main_memory':1957,
           'main_time_per_index':24, # hours
           'main_scratch':int(2000*args.num_maps_per_index),
           'merge_memory':64000,
           'merge_time':24,
           }

    if args.largemem:
        res['main_memory'] = 32000
        res['main_scratch'] = 32000

    if args.long:
        res['main_time_per_index'] = 24

    if 'CLUSTER_NAME' in os.environ:
        
        if os.environ['CLUSTER_NAME'] == 'perlmutter':
            res['pass'] = {'constraint': 'cpu', 'qos': 'shared'}
            res['main_nsimult'] = 200

        if os.environ['CLUSTER_NAME'] == 'euler':
            res['main_nsimult'] = 20
    
    return res


def main(indices, args):
    """
    Convert the format of the CosmologyGridV1.1 hdf5 files to the format of the CosmologyGridV1.0 hdf5 files.
    """

    args = setup(args)
    conf = utils_config.load_config(args.config)

    # make output
    utils_io.robust_makedirs(args.dir_out)
    utils_io.ensure_permissions(args.dir_out, verb=True)

    # change dir to temp
    tmp_dir = os.environ['TMPDIR'] if 'TMPDIR' in os.environ else os.getcwd()
    os.chdir(tmp_dir)
    LOGGER.info(f'changed dir to  {os.getcwd()}')
    if args.dir_out is None:
        args.dir_out = tmp_dir
    LOGGER.info(f'storing results in {args.dir_out}')
    
    
    # make list of grid sims to iterate over    
    ids_sim = []
    n_params = 2500
    n_reals = 7
    for i in range(n_reals):
        for j in range(n_params):
            ids_sim.append(j*n_reals+i)
    
    # loop over sims 
    for index in indices: 

        LOGGER.info(f'==================================================> index={index} num_maps_per_index={args.num_maps_per_index}')
        time_start = time.time()

        simslist_all, parslist_all, shellinfo_all = utils_cosmogrid.get_baryonified_simulations_list(conf, set_type='grid')

        
        file_out = os.path.join(args.dir_out, f'cosmogridv1_{index:05d}_nside{NSIDE_OUT}.h5')
        # create/reset file
        with h5py.File(file_out, 'w') as f:
            LOGGER.info(f'creating file {file_out}')
        
        # todo
        todo_ids = np.arange(index*args.num_maps_per_index, (index+1)*args.num_maps_per_index)
        for id_sample in todo_ids:
            append_cosmogridv1_to_file(args, conf, file_out, ids_sim[id_sample], id_sample, simslist_all, shellinfo_all)

        # if needed, copy to external archive and remove
        if args.dir_out_archive is not None:

            utils_io.archive_results(files_to_copy=[file_out],
                                     dir_out=args.dir_out,
                                     dir_out_archive=args.dir_out_archive)


        LOGGER.info(f'done with index {index} time={(time.time()-time_start)/60.:2.2f} min')

        yield index

def append_cosmogridv1_to_file(args, conf, file_out, id_sim, id_sample, simslist_all, shellinfo_all):
    """
    Append the CosmologyGridV1.1 hdf5 file to the CosmologyGridV1.0 hdf5 file.
    """

    sim_current = simslist_all[id_sim]
    shellinfo_current = shellinfo_all[sim_current['path_par']]

    LOGGER.info(f"=============================> id_sample={id_sample} id_sim={id_sim} sim={sim_current['path_par']}")

    variant = 'v11dmo'
   
    # get the right input and load
    filename_shells, _ = get_filename_shells_for_variant(variant)
    filepath_shells_local = utils_maps.copy_cosmogrid_file(conf, path_sim=sim_current['path_sim'], filename=filename_shells, check_existing=args.test, store_key='bary')
    
    sim_current = utils_arrays.rewrite(sim_current[STORE_COLS])
    
    with h5py.File(filepath_shells_local, 'r') as f_in:
    
        with h5py.File(file_out, 'a') as f_out:

            f_out.create_dataset(name=f'{id_sample:05d}/shells_info', data=shellinfo_current)
            f_out.create_dataset(name=f'{id_sample:05d}/cosmo_params', data=sim_current)
            
            list_keys = list(f_in['nobaryon_shells'].keys())
            for key in LOGGER.progressbar(list_keys, desc='loading shells'):
                shell = np.array(f_in['nobaryon_shells'][key])
                shell = hp.ud_grade(shell, nside_out=NSIDE_OUT, power=-2)
                shell = shell.astype(np.uint32)
                dset = f'{id_sample:05d}/shells/{key}'
                f_out.create_dataset(name=dset, data=shell, compression='gzip', compression_opts=5)
            
    LOGGER.info(f'cleaning up temp CosmoGridV1 files')
    utils_maps.cleanup_cosmogrid_files()

    LOGGER.info(f'done with simulation id {id_sample}')



    



def get_filename_shells_for_variant(variant):

    # for CosmoGridV1.1
    if variant in ['v11dmb', 'v11dmo']:
        filename_shells = 'baryonified_shells_v11.h5'
        nside = 1024

    else:
        raise Exception(f'unknown analysis variant {variant}')

    return filename_shells, nside

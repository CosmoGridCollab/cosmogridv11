# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""

Created July 2022
author: Tomasz Kacprzak
"""

import io, os, sys, warnings, argparse, h5py, numpy as np, time, logging, itertools, shutil, yaml
import healpy as hp
from glob import glob
from tqdm import tqdm, trange
from cosmogrid_des_y3 import utils_logging, utils_io, utils_arrays

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

cols_cosmo = ['As', 'bary_Mc', 'bary_nu', 'H0', 'O_cdm', 'O_nu', 'Ob', 'Ol', 'Om', 'm_nu', 'ns', 's8', 'w0', 'wa', 'delta', 'sobol_index', 'benchmark_type', 'id_param', 'path_par', 'box_size_Mpc_over_h', 'n_particles', 'n_shells', 'n_steps']
cols_all = ['As', 'bary_Mc', 'bary_nu', 'H0', 'O_cdm', 'O_nu', 'Ob', 'Ol', 'Om', 'm_nu', 'ns', 'pkd_seed', 's8', 'sobol_seed:i8', 'w0', 'wa', 'seed_index:i4', 'delta:U128', 'sobol_index:i4', 'benchmark_type:U128', 'id_sim:i4', 'id_param:i4', 'path_sim:U128', 'path_par:U128', 'box_size_Mpc_over_h', 'n_particles:i8', 'n_shells:i4', 'n_steps:i4']

def setup(args):

    description = 'Make metainfo for CosmoGrid'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
                        help='logging level')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    parser.add_argument('--path_cosmogrid', type=str, required=True,
                        help='path to CosmoGrid root dir (without CosmoGrid)')

    args = parser.parse_args(args)

    utils_logging.set_all_loggers_level(args.verbosity)

    return args

def main(indices, args):

    sims_benchmark, pars_benchmark = get_tables_benchmarks(args.path_cosmogrid, test=args.test)
    sims_fiducial, pars_fiducial = get_tables_fiducial(args.path_cosmogrid, test=args.test)
    sims_grid, pars_grid = get_tables_grid(args.path_cosmogrid, test=args.test)
    sims_all = np.concatenate([sims_fiducial, sims_grid, sims_benchmark])
    pars_all = np.concatenate([pars_fiducial, pars_grid, pars_benchmark])

    shell_info = get_shell_info(args.path_cosmogrid, pars_all, test=args.test)


    dict_out = {'simulations/fiducial' : sims_fiducial,
                'simulations/grid' : sims_grid,
                'simulations/benchmark' : sims_benchmark,
                'simulations/all' : sims_all,
                'parameters/fiducial' : pars_fiducial,
                'parameters/grid' : pars_grid,
                'parameters/benchmark' : pars_benchmark,
                'parameters/all' : pars_all,
                'shell_info' : shell_info}

    fname_out = 'CosmoGrid_metainfo.pkl.gz'
    utils_io.write_to_pickle(fname_out, dict_out)

def load_yaml(fname):

    with open(fname, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded
def update_rec(i, params, params_new):

    for k in params_new:
        params[k][i] = params_new[k]

def get_tables_benchmarks(path_cosmogrid, test=False):

    # this is hardcoded here to avoid opening the gz param_files.tar.gz which would take a long time
    benchmark_conf = {'box_size':            {'n_particles': 2080**3 , 'box_size_Mpc_over_h': 2250, 'n_steps': 140, 'n_shells': 69},
                      'fiducial_bench':      {'n_particles': 832**3  , 'box_size_Mpc_over_h': 900,  'n_steps': 140, 'n_shells': 69},
                      'particle_count':      {'n_particles': 2048**3 , 'box_size_Mpc_over_h': 900,  'n_steps': 140, 'n_shells': 69},
                      'redshift_resolution': {'n_particles': 832**3  , 'box_size_Mpc_over_h': 900,  'n_steps': 500, 'n_shells': 392}}

    LOGGER.info('==== getting tables for benchmark')

    filelist = glob(os.path.join(path_cosmogrid, 'CosmoGrid/raw/benchmarks/*/run_*/params.yml'))
    filelist.sort()

    if test:
        LOGGER.warning('============ TEST!')
        filelist = filelist[:50]

    LOGGER.info('found {} simulations'.format(len(filelist)))


    benchmark_type = lambda fname: str(fname.split('benchmarks/')[1].split('/run')[0])
    delta_index = lambda fname: str(fname.split('/cosmo_')[1].split('/')[0])
    seed_index = lambda fname: int(fname.split('run')[1].split('_')[1].split('/')[0])
    path_param = lambda fname: str(fname.split('run')[0].split(path_cosmogrid)[1])
    path_sim = lambda fname: str(fname.strip('params.yaml').split(path_cosmogrid)[1])

    params = utils_arrays.zeros_rec(len(filelist), columns=cols_all)
    for i, fname in LOGGER.progressbar(list(enumerate(filelist))):
        
        update_rec(i, params, params_new=load_yaml(fname))

        btype = benchmark_type(fname)

        params[i]['seed_index'] = seed_index(fname)
        params[i]['delta'] = 'none'
        params[i]['sobol_index'] = -999
        params[i]['benchmark_type'] = btype
        params[i]['id_sim'] = i
        params[i]['id_param'] = 0 # dummy
        params[i]['path_sim'] = path_sim(fname)
        params[i]['path_par'] = path_param(fname) 
        params[i]['box_size_Mpc_over_h'] = benchmark_conf[btype]['box_size_Mpc_over_h']
        params[i]['n_particles'] = benchmark_conf[btype]['n_particles']
        params[i]['n_steps'] = benchmark_conf[btype]['n_steps']
        params[i]['n_shells'] = benchmark_conf[btype]['n_shells']

    # save full grid array
    params['id_param'] = 0

    # save cosmo only
    uv, ui = np.unique(params['benchmark_type'], return_index=True)
    cosmo_params = params[np.sort(ui)][cols_cosmo]
    LOGGER.info(f'found {len(cosmo_params)} unique cosmologies')

    return params, cosmo_params


def get_tables_fiducial(path_cosmogrid, test=False):

    # hard coded here
    n_particles = 832**3
    box_size_Mpc_over_h = 900
    n_shells = 69
    n_steps = 140

    LOGGER.info('==== getting tables for fiducial')

    filelist_fiducial = glob(os.path.join(path_cosmogrid, 'CosmoGrid/raw/fiducial/cosmo_fiducial/run_*/params.yml'))
    filelist_fiducial.sort()

    filelist_delta = glob(os.path.join(path_cosmogrid, 'CosmoGrid/raw/fiducial/cosmo_delta_*/run_*/params.yml'))
    filelist_delta.sort()

    filelist = filelist_fiducial + filelist_delta

    if test:
        LOGGER.warning('============ TEST!')
        filelist = filelist[:300]

    LOGGER.info('found {} simulations'.format(len(filelist)))

    delta_index = lambda fname: str(fname.split('/cosmo_')[1].split('/')[0])
    seed_index = lambda fname: int(fname.split('run')[1].split('_')[1].split('/')[0])
    path_param = lambda fname: str(fname.split('run')[0].split(path_cosmogrid)[1])
    path_sim = lambda fname: str(fname.strip('params.yaml').split(path_cosmogrid)[1])

    params = utils_arrays.zeros_rec(len(filelist), columns=cols_all)
    for i, fname in LOGGER.progressbar(list(enumerate(filelist))):

        params_new = load_yaml(fname)
        update_rec(i, params, params_new)

        params[i]['seed_index'] = seed_index(fname)
        params[i]['delta'] = delta_index(fname)
        params[i]['sobol_index'] = -999
        params[i]['benchmark_type'] = str('none')
        params[i]['id_sim'] = i
        params[i]['id_param'] = 0 # dummy
        params[i]['path_sim'] = path_sim(fname)
        params[i]['path_par'] = path_param(fname) 
        params[i]['box_size_Mpc_over_h'] = box_size_Mpc_over_h
        params[i]['n_particles'] = n_particles
        params[i]['n_shells'] = n_shells
        params[i]['n_steps'] = n_steps

    # save full grid array
    sobol_ids = list(np.sort(np.unique(params['delta'])))
    for i in range(len(params)):
        params['id_param'][i] = sobol_ids.index(params['delta'][i])


    # sorting
    uv, ui = np.unique(params['delta'], return_index=True)
    delta_types = ['fiducial'] + list(uv[:-1])    
    params_sorted = []
    for delta in delta_types:
        select = params['delta'] == delta
        params_ = params[select]
        sorting = np.argsort(params_['seed_index'])
        params_ = params_[sorting]
        params_sorted.append(params_)
    params_sorted = np.concatenate(params_sorted)

    # save cosmo only
    uv, ui = np.unique(params_sorted['delta'], return_index=True)
    cosmo_params = params[np.sort(ui)][cols_cosmo]
    LOGGER.info(f'found {len(cosmo_params)} unique cosmologies')

    return params_sorted, cosmo_params


def get_tables_grid(path_cosmogrid, test=False):

    # hard coded here
    n_particles = 832**3
    box_size_Mpc_over_h = 900
    n_shells = 69
    n_steps = 140


    LOGGER.info('==== getting tables for grid')

    filelist = glob(os.path.join(path_cosmogrid, 'CosmoGrid/raw/grid/cosmo_*/run_*/params.yml'))
    filelist.sort()

    if test:
        LOGGER.warning('============ TEST!')
        filelist = filelist[:50]

    LOGGER.info('found {} simulations'.format(len(filelist)))

    sobol_index = lambda fname: int(fname.split('CosmoGrid/raw/grid/cosmo_')[1].split('/')[0])
    seed_index = lambda fname: int(fname.split('run')[1].split('_')[1].split('/')[0])
    path_param = lambda fname: str(fname.split('run')[0].split(path_cosmogrid)[1])
    path_sim = lambda fname: str(fname.strip('params.yaml').split(path_cosmogrid)[1])

    params = utils_arrays.zeros_rec(len(filelist), columns=cols_all)
    for i, fname in LOGGER.progressbar(list(enumerate(filelist))):

        update_rec(i, params, params_new=load_yaml(fname))

        params[i]['seed_index'] = seed_index(fname)
        params[i]['delta'] = str('none')
        params[i]['sobol_index'] = sobol_index(fname)
        params[i]['benchmark_type'] = str('none')
        params[i]['id_sim'] = i
        params[i]['id_param'] = 0 # dummy
        params[i]['path_sim'] = path_sim(fname)
        params[i]['path_par'] = path_param(fname) 
        params[i]['box_size_Mpc_over_h'] = box_size_Mpc_over_h
        params[i]['n_particles'] = n_particles
        params[i]['n_shells'] = n_shells
        params[i]['n_steps'] = n_steps

    # save full grid array
    sobol_ids = list(np.sort(np.unique(params['sobol_index'])))
    for i in range(len(params)):
        params['id_param'][i] = sobol_ids.index(params['sobol_index'][i])

    # save cosmo only
    uv, ui = np.unique(params['id_param'], return_index=True)
   
    cosmo_params = params[np.sort(ui)][cols_cosmo]
    LOGGER.info(f'found {len(cosmo_params)} unique cosmologies')

    # sorting 
    sorting = np.argsort(params['path_sim'])
    params = params[sorting]

    return params, cosmo_params


def get_shell_info(path_cosmogrid, parslist, test=False):
    
    select_cols = ['shell_id', 'lower_z', 'upper_z', 'lower_com', 'upper_com', 'shell_com']

    dict_out = {}

    if test:
        LOGGER.warning('============ TEST!')
        parslist = parslist[:500]

    LOGGER.info('getting shell_info for all sims')
    for i in LOGGER.progressbar(range(len(parslist))):

        path_par = parslist[i]['path_par']
        if '/fiducial/' in path_par:
            path_root = os.path.join(path_cosmogrid, path_par, 'run_0000')
        else:
            path_root = os.path.join(path_cosmogrid, path_par, 'run_0')

        path_sim = os.path.join(path_root, 'baryonified_shells.npz')
        path_tar = os.path.join(path_root, 'param_files.tar.gz')
    
        shells = np.load(path_sim)
        shell_info = np.array(shells['shell_info'])
        shell_info = np.sort(shell_info, order='shell_id')
        dset_name = parslist[i]['path_par']
        dict_out[dset_name] = shell_info[select_cols]

    return dict_out



if __name__=='__main__':

    args = setup(sys.argv[1:])
    main([0], args)

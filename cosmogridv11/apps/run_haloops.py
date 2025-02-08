# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""

Created July 2022
author: Tomasz Kacprzak
"""

import io, os, sys, warnings, argparse, h5py, numpy as np, time, logging, itertools, shutil, yaml, tempfile
import healpy as hp
from glob import glob
from tqdm import tqdm, trange
from cosmogridv11 import utils_logging, utils_io, utils_arrays, utils_cosmogrid, utils_config, utils_maps, utils_shells
from cosmogridv11.baryonification import utils as utils_bary
from cosmogridv11.copy_guardian import NoFileException
from cosmogridv11.filenames import *

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

def show_help():

    help_text = """
    run_haloops usage instructions
    ======================================

    Objective: perform various operations on cosmogrid shells using halo catalog

    The app is has the following commands:
    (1) profile_halos: create halocone with profiled halos
    (2) baryonify_shells: create baryonified maps
    (3) paint_haloshells: create halo maps
        
    Individual commands:

    (1) run_haloops profile_halos
    Input: halos snapshots, metadata
    Output: halocone file with profiled halos

    (2) run_haloops baryonify_shells
    Input: halocone from step (1), particle shells, metadata
    Output: shells with displaced particles

    (3) run_haloops paint_haloshells
    Input: halocone from step (1), metadata
    Output: shells with painted halos profiles (rho and rho^2)

    """
    print(help_text)


def setup(args):

    description = 'Various halo related operations'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    # parser.add_argument('command', type=str, default='help', choices=('help', 'profile_halos', 'baryonify_shells', 'paint_haloshells'), 
    parser.add_argument('command', type=str, default='help', choices=('help', 'profile_halos', 'baryonify_shells'), 
                        help='command to run, "help" for instructions')
    parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
                        help='logging level')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    parser.add_argument('--long', action='store_true',
                        help='if to run on the long queue, 7 days')
    parser.add_argument('--config', type=str, required=True, 
                        help='configuration yaml file')
    parser.add_argument('--dir_out', type=str, required=False, default=None, 
                        help='output dir for the results')
    parser.add_argument('--dir_out_archive', type=str, default=None, 
                        help='output dir for archiving the results, if specified the data will be copied there and removed from dir_out')

    args, _  = parser.parse_known_args(args)

    utils_logging.set_all_loggers_level(args.verbosity)

    args.config = utils_io.get_abs_path(args.config)
    if args.dir_out is not None:
        args.dir_out = utils_io.get_abs_path(args.dir_out)
        utils_io.robust_makedirs(args.dir_out)

    return args


def resources(args):

    if type(args) is list:
        args = setup(args)

    if args.command == 'profile_halos':
        
        main_memory = 64000 if args.long else 4000
        main_time_per_index = 4 if args.long else 24 
    
    elif args.command == 'baryonify_shells':

        main_memory = 64000 if args.long else 8000
        main_time_per_index = 4 if args.long else 24

            
    res = {'main_memory': main_memory,
           'main_time_per_index': main_time_per_index, # hours
           'main_scratch':20000,
           'merge_memory':64000,
           'main_nsimult':200,
           'merge_time':24}

    if 'CLUSTER_NAME' in os.environ:

            if os.environ['CLUSTER_NAME'].lower() == 'perlmutter':
                res['pass'] = {'constraint': 'cpu', 'account': 'des', 'qos': 'shared'}

            elif os.environ['CLUSTER_NAME'].lower() == 'euler':

                if args.command == 'baryonify_shells':
                    
                    res['main_nsimult'] = 400
                
    
    return res

def print_baryon_params(params_bary):

    LOGGER.info('baryon params set:')
    for c in dir(params_bary.baryon):
        if not c.startswith('_'):
            LOGGER.info(f'{c:>20s} = {str(getattr(params_bary.baryon, c)):<50s}')

def print_sim_params(sim):

    LOGGER.debug('simulation params:')
    for c in sim.dtype.names: 
        LOGGER.info(f'{c:>20s} = {str(sim[c]):<50s}')

def load_sim_baryparams(filename_params_local, sim):

    utils_cosmogrid.decompress_params_files(filename_params_local)
    filename_baryparams = os.path.join(os.getcwd(), 'baryonification_params.py')
    params_bary = utils_bary.load_baryonification_params(filename_baryparams)

    pars = ['Mc', 'eta_cga', 'eta_tot', 'mu', 'nu', 'thco', 'thej']
    for p in pars:
        setattr(params_bary.baryon, p, sim[f'bary_{p}'])
    
    print_baryon_params(params_bary)
    
    return params_bary


# @profile
def main(indices, args):

    args = setup(args)
    conf = utils_config.load_config(args.config)

    simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_baryonified_simulations_list(conf, set_type='all')

    tmp_dir = os.environ['TMPDIR'] if 'TMPDIR' in os.environ else os.getcwd()
    os.chdir(tmp_dir)
    LOGGER.info(f'changed dir to  {os.getcwd()}')

    if args.dir_out is None:
        args.dir_out = os.getcwd()

    if args.command == 'help':

        show_help()
        sys.exit(0)

    for index in indices:

        sim_current = simslist_all[index]
        path_sim_bary = sim_current['path_sim']
        path_par_bary = sim_current['path_par']
        path_sim_raw = path_sim_bary.replace('CosmoGrid/bary/', f'CosmoGrid/raw/')
        path_par_raw = path_par_bary.replace('CosmoGrid/bary/', f'CosmoGrid/raw/')
        shell_info = shell_info_all[path_par_bary]
        dir_out_current = os.path.join(args.dir_out, path_sim_bary)
        utils_io.robust_makedirs(dir_out_current)

        print_sim_params(sim_current)
        

        if args.command == 'profile_halos':

            filename_halocone_out =  get_filename_profiled_halos(dir_out=dir_out_current, tag=conf['baryonification']['tag'])

            # for delta simulations with deltas in baryon parameter, just copy the fiducial
            if 'delta_bary' in sim_current['delta']:
                
                copy_corresponding_fiducial_halocone(args, conf, index, simslist_all, filename_halocone_out)

            else:

                filename_halos_local, filename_params_local = utils_maps.copy_cosmogrid_file(conf, path_sim_raw, 
                                                                                             filename=['pkd_halos.tar.gz', 'param_files.tar.gz'], 
                                                                                             check_existing=True, 
                                                                                             noguard=args.test)

                utils_maps.decompress_halos_files(filename_halos_local)
                params_bary = load_sim_baryparams(filename_params_local, sim_current)

                # main magic - get the halos
                store_halocone(conf,  shell_info, params_bary,
                               test=args.test,
                               filename_out=filename_halocone_out)

                # if needed, copy to external archive and remove
                if args.dir_out_archive is not None:

                    utils_io.archive_results(files_to_copy=[filename_halocone_out],
                                             dir_out=args.dir_out,
                                             dir_out_archive=args.dir_out_archive)



        elif args.command == 'baryonify_shells':

            if 'delta_bary' in sim_current['delta']:
    
                # replace delta_bary_* params with fiducial
                path_sim_raw = path_sim_raw.replace(sim_current['delta'], 'fiducial')
                LOGGER.info(f'this is a delta_bary simulation, using maps from cosmo_fiducial {path_sim_raw}')

            # copy files and load parameters
            filename_params_local, filename_shells_local, filename_halos_local = utils_maps.copy_cosmogrid_file(conf, path_sim_raw, 
                                                                                                                filename=['param_files.tar.gz', 
                                                                                                                           get_filename_compressed_shells(path_sim_raw), 
                                                                                                                           get_filename_profiled_halos('', tag=conf['baryonification']['tag'])], 
                                                                                                                check_existing=True, 
                                                                                                                store_key=['root', 'root', 'halos'],
                                                                                                                noguard=args.test)
            # main magic - baryonify shells
            filename_shells_out =  get_filename_baryonified_shells(dir_out=dir_out_current, tag=conf['baryonification']['tag'])
            baryonify_shells(conf, filename_halos_local, filename_shells_local, filename_shells_out,
                             param=load_sim_baryparams(filename_params_local, sim_current), 
                             sim_current=sim_current,
                             test=args.test)

            # if needed, copy to external archive and remove
            if args.dir_out_archive is not None:

                utils_io.archive_results(files_to_copy=[filename_shells_out],
                                         dir_out=args.dir_out,
                                         dir_out_archive=args.dir_out_archive)

        yield index

    

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


def create_shell_table(cosmo, shell_z):

    from cosmogridv11.baryonification.halo_utils import get_shell_cov_dist

    # a list for the shell file
    shell_dtype = np.dtype([("shell_id", np.uint16),
                            ("lower_z", np.float32), 
                            ("upper_z", np.float32),
                            ("lower_com", np.float32), 
                            ("upper_com", np.float32), 
                            ("shell_com", np.float32)])

    shell_tab = np.zeros(len(shell_z), dtype=shell_dtype)

    for i_file, z_boundary in enumerate(shell_z):

        shell_cov, cov_inner, cov_outer = get_shell_cov_dist(z_boundary, cosmo)
        shell_tab[i_file] = i_file, z_boundary[0], z_boundary[1], cov_inner, cov_outer, shell_cov

    return shell_tab




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



######################################################################################
######################################################################################
##
## Halo model conversions and profiling
##
######################################################################################
######################################################################################


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


def store_halocone(conf, shell_info, params_bary, filename_out, test=False):

    """
    https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/carpet/baryons/gen_halocone.py
    This file creates a HaloCone from a CosmoGrid simulation.
    """

    from cosmogridv11 import baryonification

    # get redshift bounds
    z_bounds = np.stack([shell_info['lower_z'], shell_info['upper_z']], axis=1)

    # match the corresponding HaloFiles
    halo_files = baryonification.utils.get_halofiles_from_z_boundary(z_boundaries=z_bounds, 
                                                                     log_file="CosmoML.log", 
                                                                     prefix="./pkd_halos/CosmoML")


    LOGGER.info("Extracted redshift bounds and matched halofiles:".format(z_bounds))
    for z, h in zip(z_bounds, halo_files):
        print(z, h)

    if conf['baryonification']['tag'] == 'v1':

        LOGGER.info('using legacy baryonification from CosmoGridV1, make sure to test it before use')

        raise Exception('TODO: implement storing results for each shell separately for the v1 legacy pipeline')

    elif conf['baryonification']['tag'] == 'v11':
        
        LOGGER.info('using baryonification CosmoGridV1.1')

        store_v11_halos(param=params_bary,
                        halo_files=halo_files,
                        shells_z=z_bounds,
                        filename_out=filename_out, 
                        test=test)
 
    
def store_v11_halos(param, halo_files, shells_z, filename_out, max_repli=7, test=False):
    """
    Creates a halo file for the lightcone, given the halo files of the snapshots and their cut out shells
    :param param: baryonification params
    :param halo_files: list of halo files
    :param shells_z: An array containing the redshifts of the shells with shape [n_halo_files, 2], where the first
                     column is the lower bound and the second column represents the upper bound
    :param output_path: path where to save the file
    :param max_repli: maximum number of replicates
    :param out_type: currently only 'AHF-NUMPY'
    :return: name of the written file
    """

    from cosmogridv11.baryonification import halo_utils

    # check for consitency
    assert len(halo_files) == len(shells_z), "Halo file and boundaries need to match in first dimension"

    # get a cosmology object
    cosmo = halo_utils.build_cosmo(param=param)
    rho_c0 = halo_utils.get_rho_c_for_z(cosmo, z=0)
    part_mass = halo_utils.get_particle_mass(cosmo, Lbox=param.sim.Lbox, nparts=param.sim.nparts)

    # get shell table
    shell_tab = create_shell_table(cosmo, shells_z)

    # save the relevant lines and shell_data
    LOGGER.info(f'storing in {filename_out}')
    for i, (halo_, shell_) in enumerate(zip(halo_files, shell_tab)):

        LOGGER.info(f'=============== preparing shell {i+1}/{len(halo_files)}')

        # get halo lightcone for this shell
        h_shell = get_v11_halos_for_shell(cosmo, param, i, halo_, shell_, part_mass, rho_c0, test=test)

        # # remove infeasable shell ids:
        select = h_shell['shell_id'] < len(shell_tab)
        h_shell = h_shell[select]
        h_buffer = np.concatenate([h_buffer, h_shell]) if i>0 else h_shell

        # store previous shell
        if i > 0:

            select_store = h_buffer['shell_id'] == i-1
            store_profiled_halos(filename_out, h_buffer[select_store], shell_tab_prev, param, tag='v11', shell_id=i-1)
            shell_ids = str(np.unique(h_buffer[select_store]['shell_id']))
            halo_buffs = str(np.unique(h_buffer[select_store]['halo_buffer']))
            LOGGER.info(f"stored halos at shell_id={i-1} shell_ids={shell_ids} halo_buffers={halo_buffs}")
            h_buffer = h_buffer[~select_store]

        # store current shell if it's the last
        if i+1 == len(shell_tab):

            select_store = h_buffer['shell_id'] == i
            store_profiled_halos(filename_out, h_buffer[select_store], shell_tab, param, tag='v11', shell_id=i)
            shell_ids = str(np.unique(h_buffer[select_store]['shell_id']))
            halo_buffs = str(np.unique(h_buffer[select_store]['halo_buffer']))
            LOGGER.info(f"stored halos at shell_id={i} shell_ids={shell_ids} halo_buffers={halo_buffs}")

        shell_tab_prev = shell_tab

    LOGGER.info(f'finished writing to {filename_out}')



def get_v11_halos_for_shell(cosmo, param, ind_shell, halos, shell, part_mass, rho_c0, max_repli=7, test=False):

    from cosmogridv11.baryonification import halo_utils

    # get basic parameters
    z_shell = (shell['lower_z']+shell['upper_z'])/2.
    rho_c = halo_utils.get_rho_c_for_z(cosmo, z_shell)
    
    # get the halos
    h = halo_utils.read_pkd_halos(file_name=halos, count=10000 if test else -1)
    h = halo_utils.dequantize_pkd_halos(halo_data=h, Lbox=param.sim.Lbox, rho_c0=rho_c0, part_mass=part_mass)
    cols_select = ['mass', 'x', 'y', 'z']
    h = utils_arrays.rewrite(h[cols_select])
    h = halo_utils.add_halo_id(h, ind_shell)

    # box replicate 
    h_shell = halo_utils.box_replicate_shell_halos_to_lightcone(h, param, 
                                                                cov_lims=[shell['lower_com'], shell['upper_com']], 
                                                                max_repli=max_repli,
                                                                i_file=ind_shell)

    # add columns for the halo profile parameters
    h_shell = utils_arrays.add_cols(h_shell, names=['m_200c:f8', 'r_200c:f4', 'c_200c:f4'])

    # calculate 200c parameters
    if len(h_shell)==0:

        LOGGER.info('shell {} z={:4.3f} no halos found'.format(ind_shell+1, z_shell))

    else:

        # convert FoF masses to 200c halos for baryonification
        from hmf.halos import mass_definitions as md
        mass_def_200c = md.SOCritical(overdensity=200)
        mass_def_fof = md.FOF(linking_length=0.2)

        # this uses Duffy08 for m-c relation
        # https://github.com/halomod/hmf/blob/75459311ee78d58abf4c5afc4163b632ebb8fdfd/src/hmf/halos/mass_definitions.py#L156
        # now profile only the selected halos
        uv, uind, uinv = np.unique(h_shell['ID'], return_index=True, return_inverse=True)
        ensure1d = lambda args: [np.atleast_1d(a) for a in args]
        m_200c, r_200c, c_200c = ensure1d(mass_def_fof.change_definition(m=h_shell[uind]['mass'], mdef=mass_def_200c, cosmo=cosmo))
        LOGGER.info('shell {} z={:4.3f} selected {}/{} replicated halos unique {}'.format(ind_shell+1, z_shell, len(h_shell), len(h), len(uv)))
        LOGGER.info("converted {} halos from fof to 200c, m_200c: [{:1.4e}, {:1.4e}], r_200c: [{:2.4f}, {:2.4f}], c_200c: [{:2.4f}, {:2.4f}]".format(
            len(uv), np.min(m_200c), np.max(m_200c), np.min(r_200c), np.max(r_200c), np.min(c_200c), np.max(c_200c)))

        # h_shell = utils_arrays.add_cols(h_shell, names=['m_200c:f8', 'r_200c:f4', 'c_200c:f4'])
        h_shell['m_200c'] = m_200c[uinv]
        h_shell['r_200c'] = r_200c[uinv]
        h_shell['c_200c'] = c_200c[uinv]

    return h_shell




def store_profiled_halos(filename_out, halo_data, shell_data, param, tag, shell_id=0):

    cols_lightcone, cols_halosprop = get_halo_cols_to_store(tag)
    compression_args = dict(compression='gzip', shuffle=True, compression_opts=5)
    mode = 'w' if shell_id==0 else 'a'

    if halo_data is None:

        num_halos = 0
        halo_props = h5py.Empty('f')
        halo_pos = h5py.Empty('f')

    else:

        # to save storage, only save [x, y, z, ids..] for all halos on the lightcone
        # the other parameters are stored onece per unique halo ID
        # cols_base = list(set(halo_data.dtype.names)-set(cols_lightcone))
        uval, uind, uinv = np.unique(halo_data['ID'], return_index=True, return_inverse=True)
        halo_props = utils_arrays.rewrite(halo_data[cols_halosprop]) # use only base columns
        halo_props = halo_props[uind] # select unique halos
        halo_pos = utils_arrays.rewrite(halo_data[cols_lightcone]) # use only lightcone columns
        halo_pos = utils_arrays.add_cols(halo_pos, names=['uid:u8'], data=[uinv]) # add unique id
        num_halos = len(halo_data)

    with h5py.File(filename_out, mode) as f:

        f.create_dataset(name=f'shell{shell_id:03d}/halo_props', data=halo_props, **compression_args)
        f.create_dataset(name=f'shell{shell_id:03d}/halo_pos', data=halo_pos, **compression_args)
        f[f'shell{shell_id:03d}/halo_props'].attrs['MinParts'] = param.sim.Nmin_per_halo

        # store shell data
        if 'shell_data' not in f.keys():
            f.create_dataset(name=f'shell_data', data=shell_data, **compression_args)
    
    LOGGER.info(f'stored shell_id={shell_id} num_halos={num_halos} with columns {str(cols_lightcone+cols_halosprop)}')


def read_profiled_halos(filename, shell_id='all', add_xyz_shell=True):

    # load the halos
    with h5py.File(filename) as f:

        shells = np.array(f["shell_data"])

        if shell_id == 'all':

            stack_key = lambda n: np.concatenate([np.array(f[f"shell{i:03d}/{n}"]) for i in range(len(shells))])
            halo_pos = stack_key('halo_pos')
            halo_pos = stack_key('halo_props')

        else:

            halo_pos = np.array(f[f"shell{shell_id:03d}/halo_pos"])
            halo_props = np.array(f[f"shell{shell_id:03d}/halo_props"])

    halos = halo_props[halo_pos['uid']]
    halos = utils_arrays.merge_recs([halo_pos, halos])

    if add_xyz_shell:

        # we project the halo coordinates onto the shell they fall in
        norm_kPc = np.sqrt(halos['x'] ** 2 + halos['y'] ** 2 + halos['z'] ** 2)
        shell_cov = shells["shell_com"][halos["shell_id"]]
        shell_cov_kpc = shell_cov * 1000.
        halos = utils_arrays.add_cols(halos, names=['x_shell:f4', 'y_shell:f4', 'z_shell:f4'])
        halos['x_shell'] = halos['x'] * shell_cov_kpc / norm_kPc
        halos['y_shell'] = halos['y'] * shell_cov_kpc / norm_kPc
        halos['z_shell'] = halos['z'] * shell_cov_kpc / norm_kPc

        LOGGER.info('read {} halos from {}, shell_id={}'.format(len(halos), filename,  shell_id))

    return halos


def get_halo_cols_to_store(tag):

    cols_halosprop = ['ID']
    cols_lightcone = ['x', 'y', 'z', 'shell_id', 'halo_buffer']

    if tag == 'v1':

        cols_halosprop += ['profile_Mvir', 'profile_Nvir', 'profile_rvir', 'profile_cvir', 'profile_success']

    elif tag == 'v11':

        cols_halosprop += ['m_200c', 'r_200c', 'c_200c']

    return cols_lightcone, cols_halosprop



######################################################################################
######################################################################################
##
## Shell painting
##
######################################################################################
######################################################################################


def create_haloshells(conf, halos, cosmo, n_shells, test=False):

    shell_ids = np.unique(halos['shell_id']).astype(int)
    if test:
        shell_ids = np.array([shell_ids[20]])
        LOGGER.warning('===============================> test! using a single shell')

    nside_out = conf['baryonification']['nside_out']
    n_pix = hp.nside2npix(nside_out)
    LOGGER.info(f'making haloshells for shell_ids {str(shell_ids[[0]])} - {str(shell_ids[[-1]])}, nside={nside_out}')

    haloshells = {}
    for i, shell_id in enumerate(shell_ids):

        select = (halos['shell_id'] == shell_id) & (halos['halo_buffer']==0) 
        if 'profile_success'in halos.dtype.names:
            select &= (halos['profile_success']>0) 

        halos_current = halos[select]

        LOGGER.info(f'============= shell {i+1}/{len(shell_ids)} id={shell_id} num_halos={len(halos_current)}')

        if len(halos_current)==0:
            LOGGER.info('no halos found in this shell')
            continue

        shell_current = paint_shell(halos_current, cosmo, tag=conf['baryonification']['tag'], nside=nside_out, test=test)

        for key in shell_current.keys():
            haloshells.setdefault(key, np.zeros((n_shells, n_pix), dtype=np.float64))
            haloshells[key][shell_id] = shell_current[key]

    return haloshells


def store_haloshells(filename_out, haloshells, dtype=np.float32):

    with h5py.File(filename_out, 'w') as f:
        for k, m in haloshells.items():
            f.create_dataset(name=k, data=haloshells[k].astype(dtype), compression='gzip', compression_opts=5, shuffle=True)

    LOGGER.info(f'stored {filename_out} with datasets {list(haloshells.keys())}')


def paint_shell(halos, cosmo, nside, tag='v11', R_times=5, test=False):

    import astropy.cosmology
    astropy.cosmology.current_cosmo = cosmo
    from astropaint.profiles import NFW

    # convert PKDGrav3 halos to the catalog format used by astropaint
    catalog = get_astropaint_catalog(halos, tag=tag)

    # create templates to integrate
    templates = [NFW.rho_2D, NFW.rho_squared_2D]
    templates_interp = [NFW.rho_2D_cachedinterp, NFW.rho_2D_squared_cachedinterp]
    templates_labels = ['rho', 'rho_sq']

    # set interpolator traning grid density
    nr = 20000
    nrs = 50

    if test:
        LOGGER.warning('-----------------> test! using fewer interpolation points')
        nr = 1000
        nrs = 20

    # limits for radius r and scale radius rs
    lims_r = [5e-5, 20]
    lims_rs = [0.01, 2]
    in_lims = lambda x, lims: (x>=lims[0]) & (x<=lims[1])
    assert np.any(in_lims(catalog.data.R_s, lims_rs)), 'for some halos scale radius is outside outside interpolation range'

    haloshells = {}
    for i, (templ, templ_interp, templ_lab) in enumerate(zip(templates, templates_interp, templates_labels)):

        interp_key = f'interp_{templ_lab}_2D__{str(cosmo)}'
        if interp_key not in NFW.cache.keys(): 

            LOGGER.info(f'building interp {interp_key} lims_rs={lims_rs} lims_r={lims_r}')
            NFW.cache[interp_key] = build_r_rs_interp(templ, lims_r=lims_r, lims_rs=lims_rs, nr=nr, nrs=nrs)

        LOGGER.info(f'creating haloshell for {templ_lab}')
        haloshells[templ_lab] = get_painted_shell(catalog, nside, templ_interp, R_times=R_times)

    return haloshells



def build_r_rs_interp(f, lims_r, lims_rs, nr=4000, nrs=20):


    r = np.concatenate([np.array([1e-6,]), np.logspace(*np.log10(lims_r), nr)])
    rs = np.linspace(*lims_rs, nrs)
    y = np.zeros((len(rs), len(r)))
    for i, rs_ in LOGGER.progressbar(list(enumerate(rs)), desc=f'building f(r,rs) interpolator for f={f.__name__}'):
        y[i] = np.log(f(r, 1., rs_))
    grid_r, grid_rs = np.meshgrid(np.log(r), rs)
    points = np.vstack((grid_rs.ravel(), grid_r.ravel())).T
    values = y.ravel()
    select = np.isfinite(values)
    values = values[select]
    points = points[select]

    from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator 
    interp = LinearNDInterpolator(points, values, rescale=False, fill_value=np.nan)
    LOGGER.info(f'created LinearNDInterpolator using {len(values)}/{len(select)} points')

    return interp


######################################################################################
######################################################################################
##
## Baryonification
##
######################################################################################
######################################################################################

def convert_Mpc_to_Gpc(rec, fields):

    for f in fields:
        rec[f] = rec[f]/1e3

    return rec

def convert_kpc_to_Gpc(rec, fields):

    for f in fields:
        rec[f] = rec[f]/1e6

    return rec

def convert_kpc_to_Mpc(rec, fields):

    for f in fields:
        rec[f] = rec[f]/1e3

    return rec


def convert_halo_col_names(halos, conf):

    if conf['baryonification']['tag'] == 'v1':

        # select the halos with successful profiling
        select_success = halos['profile_success']==True
        halos = halos[select_success]
        LOGGER.info(f'using {np.count_nonzero(select_success)}/{len(select_success)} halos with successfully measured profiles')

        halos = utils_arrays.rename_cols(halos, src=list(cols_match.values()), dst=list(cols_match.keys()))
        cols_match = {'nfw_c':'profile_nfw_c', 'nfw_r':'profile_nfw_r', 'nfw_M':'profile_nfw_M'}
        halos = utils_arrays.rename_cols(halos, src=list(cols_match.values()), dst=list(cols_match.keys()))

    elif conf['baryonification']['tag'] == 'v11':

        cols_match = {'nfw_c':'c_200c', 'nfw_r':'r_200c', 'nfw_M':'m_200c'}
        halos = utils_arrays.rename_cols(halos, src=list(cols_match.values()), dst=list(cols_match.keys()))

    else:
        raise Exception('unknown baryonification tag {}'.format(conf['baryonification']['tag']))

    return halos



# @profile
def baryonify_shells(conf, filename_halocone, filename_shells, filename_shells_out, param, sim_current, test=False, delta_shell=False):
    """
    https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/carpet/baryons/baryonify_shells.py
    This runs baryonification.
    """


    """
    This script baryonifies the compressed shells of a CosmoGrid simulation given a HaloCone and parameters
    """

    from cosmogridv1.baryonification import halo_utils

    # decide the type of baryonification to perform
    use_mc_relation = True if 'v11' in conf['baryonification']['tag'] else False

    # get various config parameters
    nside_out=int(conf['baryonification']['nside_out'])
    shells_info = utils_maps.read_shell_info(filename_shells)
    filename_shells_decompressed = utils_maps.decompress_particle_shells(filename_shells)

    # we get the cosmo data
    cosmo_data = halo_utils.get_cosmo_data(param=param)

    if test:
        shells_info = shells_info[[0]]
        LOGGER.warning('------------------> test! using single shell')

    for j, shell_info in LOGGER.progressbar(list(enumerate(shells_info)), desc='displacing shells', at_level='warning'):

        shell_info = shells_info[j]
        i = shell_info['shell_id']

        LOGGER.info(f"======> shell={i+1}/{len(shells_info)} z={shell_info['lower_z']:2.3f}-{shell_info['upper_z']:2.3f}")

        # read shells and halos
        particle_shell = utils_maps.read_particle_shells(filename_shells_decompressed, shell_id=i).ravel()
        halos = read_profiled_halos(filename_halocone, shell_id=i)
        halos = convert_halo_col_names(halos, conf)
        halos = convert_kpc_to_Gpc(halos, fields=['x', 'y', 'z'])
        halos = convert_kpc_to_Mpc(halos, fields=['x_shell', 'y_shell', 'z_shell'])

        if len(halos)==0:

            LOGGER.info(f'displacing shell {i:d} no halos')

            baryon_shell = particle_shell.astype(np.float32)

            if nside_out is not None:
                baryon_shell = hp.ud_grade(baryon_shell, nside_out=nside_out, power=-2)

            shell_stats = dict(n_halos=0, n_halos_unique=0, n_parts=0, n_parts_unique=0, shell_path=shell_info['shell_name'])
        
        else:

            LOGGER.info(f'displacing shell {i:d} with n_halos={len(halos)}')

            shell_info, shell_stats, baryon_shell = halo_utils.displace_shell(params=param, 
                                                                              shell_particles=particle_shell, 
                                                                              particle_shell_info=shell_info, 
                                                                              halos=halos, 
                                                                              cosmo_data=cosmo_data, 
                                                                              delta_shell=delta_shell, 
                                                                              nside_out=nside_out,
                                                                              using_mc_relation=use_mc_relation)

        store_baryonified_shell(filename_shells_out, baryon_shell, particle_shell, sim_current, shell_id=i)


def store_baryonified_shell(filename_out, baryon_shell, particle_shell, sim_current, shell_id):

    nside_out = hp.npix2nside(len(baryon_shell))
    nside_in = hp.npix2nside(len(particle_shell))

    LOGGER.info(f'storing output {nside_in} -> {nside_out} particle_shell=[{particle_shell.shape}, {particle_shell.dtype}]  baryon_shells=[{baryon_shell.shape} {baryon_shell.dtype}]')

    orig_shell = hp.ud_grade(particle_shell, nside_out=nside_out, power=-2)
    LOGGER.info(f'orig shell {orig_shell.shape} {orig_shell.dtype} value lims=[{np.min(orig_shell)}, {np.max(orig_shell)}]')

    # convert to diff
    diff_shell = baryon_shell - orig_shell
    LOGGER.info(f'diff shell {diff_shell.shape} {diff_shell.dtype} diff stats: min={np.min(diff_shell):2.4e} max={np.max(diff_shell):2.4e} mean={np.mean(diff_shell):2.4e} median={np.median(diff_shell):2.4e}')

    diff_shell_inds = np.uint32(np.nonzero(diff_shell.ravel())[0])
    diff_shell_vals = diff_shell.ravel()[diff_shell_inds]

    compression_args = dict(compression='gzip', shuffle=True, compression_opts=4)
    with h5py.File(filename_out, 'w' if shell_id==0 else 'a') as f:
        f.create_dataset(name=f'nobaryon_shells/shell{shell_id:03d}', data=orig_shell,       **compression_args)
        f.create_dataset(name=f'diff_shell_inds/shell{shell_id:03d}', data=diff_shell_inds,  **compression_args)
        f.create_dataset(name=f'diff_shell_vals/shell{shell_id:03d}', data=diff_shell_vals,  **compression_args)
        if 'sim_params' not in f.keys():
            f.create_dataset(name='sim_params', data=utils_arrays.unicode_to_ascii(sim_current), **compression_args)

    LOGGER.info(f'stored {filename_out}')




def store_baryonified_shells(filename_out, baryon_shells, particle_shells, shell_infos_arr):

    nside_out = hp.npix2nside(baryon_shells.shape[1])
    nside_in = hp.npix2nside(particle_shells.shape[1])

    LOGGER.info(f'storing output {nside_in} -> {nside_out} particle_shells=[{particle_shells.shape}, {particle_shells.dtype}]  baryon_shells=[{baryon_shells.shape} {baryon_shells.dtype}]')

    # orig_shell = hp.ud_grade(particle_shells.astype(np.uint32), nside_out=nside_out, power=-2)
    orig_shell = particle_shells if nside_out == nside_in else np.array([hp.ud_grade(p.astype(np.uint32), nside_out=nside_out, power=-2) for p in particle_shells])
    LOGGER.info(f'orig shell {orig_shell.shape} {orig_shell.dtype} value lims=[{np.min(orig_shell)}, {np.max(orig_shell)}]')

    # convert to diff
    for i in range(len(baryon_shells)):
        baryon_shells[i] -= orig_shell[i]
    diff_shell = baryon_shells
    LOGGER.info(f'diff shell {diff_shell.shape} {diff_shell.dtype}')

    LOGGER.info(f'storing {filename_out} {diff_shell.nbytes/1e9:4.2f} GB')

    diff_shell_inds = np.uint32(np.nonzero(diff_shell.ravel())[0])
    diff_shell_vals = diff_shell.ravel()[diff_shell_inds]

    LOGGER.info(f'compressing arrays:')
    LOGGER.info('orig_shell={:4.2f} GB'.format(orig_shell.nbytes/1e9) )
    LOGGER.info('diff_shell_inds={:4.2f} GB'.format(diff_shell_inds.nbytes/1e9) )
    LOGGER.info('diff_shell_vals={:4.2f} GB'.format(diff_shell_vals.nbytes/1e9) )
    LOGGER.info('shell_infos_arr={:4.2f} GB'.format(shell_infos_arr.nbytes/1e9) )

    time_start = time.time()
    compression_args = dict(compression='gzip', shuffle=True, compression_opts=4)
    with h5py.File(filename_out, 'w') as f:
        f.create_dataset(name='nobaryon_shells', data=orig_shell,      **compression_args)
        f.create_dataset(name='diff_shell_inds', data=diff_shell_inds, **compression_args)
        f.create_dataset(name='diff_shell_vals', data=diff_shell_vals, **compression_args)
        f.create_dataset(name='shell_dicts',     data=shell_infos_arr, **compression_args)

    LOGGER.info(f'stored {filename_out}, compression time {time.time()-time_start:4.0f} sec')


def create_baryonification_info_file(fname_out, shell_infos, param):

    # accumulate stats
    from cosmogridv11.baryonification.utils import bunch_to_lists

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


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, default='[0]')
    args, args_remaining = parser.parse_known_args(sys.argv[1:])

    next(main(indices=get_indices(args.tasks), args=args_remaining))


## code graveyard

        # if args.command == 'paint_haloshells':


        #     filename_halocone =  get_filename_profiled_halos(dir_out='.', tag=conf['baryonification']['tag'])
        #     filename_halocone_local = utils_maps.copy_cosmogrid_file(conf=conf, path_sim=path_sim, filename=filename_halocone, check_existing=True, bary=True)

        #     halos = read_profiled_halos(filename_halocone_local) 

        #     LOGGER.info(f'paint_haloshells: read {len(halos)} halos from {filename_halocone_local}')

        #     haloshells = create_haloshells(conf, halos, cosmo, n_shells=len(shell_info), test=args.test)

        #     filename_haloshells_out = get_filepath_haloshells(dir_out=dir_out_current, tag=conf['baryonification']['tag'])
        #     store_haloshells(filename_haloshells_out, haloshells)

        #     # if needed, copy to external archive and remove
        #     if args.dir_out_archive is not None:

        #         utils_io.archive_results(files_to_copy=[filename_haloshells_out],
        #                                  dir_out=args.dir_out,
        #                                  dir_out_archive=args.dir_out_archive)


            # filename_shells_local = utils_maps.copy_cosmogrid_file(conf=conf, path_sim=path_sim, filename=get_filename_compressed_shells(path_sim), check_existing=True, store_key='root')
            # filename_halos_local = utils_maps.copy_cosmogrid_file(conf=conf, path_sim=path_sim, filename=get_filename_profiled_halos('', tag=conf['baryonification']['tag']), check_existing=True, store_key='halos')


# def read_shells(filename_shells):

#     with np.load(filename_shells) as f:

#         LOGGER.info("Loading shells...")
#         particle_shells = np.array(f['shells'])
#         particle_shells_info = np.array(f['shell_info'])
#         LOGGER.info("Done...")

#     return particle_shells, particle_shells_info



    # initialise parameters
    # sys.path.append('.') 
    # sys.path.append(os.path.join(baryonification.__path__[0], '..')) # this is for back-compat with baryonification_params.py scripts
    # par = load_baryonification_params(filename_params)
    # import baryonification_params
    
    # from baryonification_params import par



    # if args.command == 'paint_haloshells':

    #     res['main_nsimult'] = 1000



    # elif args.command == 'paint_haloshells':

    #     main_memory = 128000 if args.long else 32000


        # # write
        # baryonification.prep_lightcone_halos(param=params_bary,
        #                                      halo_files=halo_files,
        #                                      shells_z=z_bounds,
        #                                      filename_out=filename_out, 
        #                                      test=test)
    


    #     import pudb; pudb.set_trace();
    #     pass

    #     shells_stats.append(shell_stats)
    #     shells_back.append(shell_info)
    #     baryon_shells.append(baryon_shell)
    #     LOGGER.debug("Added shell {} to the halocone...".format(shell_stats["shell_path"]))

    #     if test:
    #         LOGGER.warning('----------> testing! skipping other baryon shells')
    #         break
                
    # # this adds shells which did not have any baryonification applied
    # # if delta_shell is False we want to add the shells without halos as well
    # if not delta_shell:
        
    #     for displ_shell, shell_info in LOGGER.progressbar(list(zip(particle_shells, particle_shells_info)), desc='adding unbaryonified shells', at_level='info'):
        
    #         if shell_info not in shells_back:
        
    #             # get the shell path
    #             shell_path = shell_info["shell_name"]

    #             # load and downsample
    #             if nside_out is not None:
    #                 if nside_out != hp.npix2nside(len(displ_shell)):
    #                     displ_shell = hp.ud_grade(displ_shell.astype(np.float32), nside_out=nside_out, power=-2)

    #             # create a shell collection
    #             shell_stats = dict(n_halos=0,
    #                                n_halos_unique=0,
    #                                n_parts=0,
    #                                n_parts_unique=0,
    #                                shell_path=shell_path)

    #             # append
    #             shells_stats.append(shell_stats)
    #             shells_back.append(shell_info)
    #             baryon_shells.append(displ_shell)


    # shells_stats_arr = halo_utils.shell_stats_list_to_arr(shells_stats)
    # shells_back = np.array(shells_back)
    # sorting = np.argsort(shells_back['shell_id'])
    # shells_back = shells_back[sorting]
    # shells_stats_arr = shells_stats_arr[sorting]
    # baryon_shells = np.array([baryon_shells[i] for i in sorting], dtype=np.float32)

    # sort according to the shell id
    # baryon_shells = baryon_shells[sorting]

    # return baryon_shells, particle_shells, shells_stats_arr



    # get limits for halo sizes and integration ranges
    # lims_rs = np.min(catalog.data['R_s'])*0.99, np.max(catalog.data['R_s'])*1.01 
    # lims_r = 1e-4, np.max(catalog.data['R_200c'])*R_times*1.01


            # np.concatenate([np.array(f[f"shell{i:03d}/halo_pos"]) for i in range(len(shells))])
            # halo_props = np.array(f["halo_props"]) fo
            # halos_MinParts = f["halo_props"].attrs['MinParts']

    # LOGGER.info(f'R_s min {np.min(catalog.data.R_s):1.4e} max {np.max(catalog.data.R_s):1.4e}')

        # NFW.cache[f'interp_{tag}_2D'] = build_r_rs_interp(templ, lims_r, lims_rs)


# def get_astropaint_catalog_v1(halos):

#     import pandas as pd
#     from astropaint import Catalog
#     from astropy.cosmology import current_cosmo as cosmo
#     from astropaint import transform
#     from astropy.coordinates import cartesian_to_spherical

#     dtype = {"names": ["x", "y", "z", "v_x", "v_y", "v_z", "R_ang_200c", "R_s", "rho_s", "theta", "phi"], "formats": 11 * [np.float32]}
#     d =  pd.DataFrame(np.zeros(len(halos), dtype))

#     # halos in kpc/h
#     # d in Mpc 
#     # convert_pos = lambda c: c / 1000 / (cosmo.H0.value/100.)
#     # convert_mass = lambda m: m / (cosmo.H0.value/100.)

#     # convert to Mpc and M_sun units
#     convert_position = lambda c: c / 1000 / (cosmo.H0.value/100.) # position in kpc/h -> Mpc
#     convert_mass = lambda m: m / (cosmo.H0.value/100.) # mass M_sun/h -> M_sun
#     convert_radius = lambda r: r / 1000 / (cosmo.H0.value/100.) # radius kpc/h -> Mpc
#     for c in ['x', 'y', 'z']: d[c] = convert_position(halos[c])
#     d['R_200c'] = convert_radius(halos['profile_nfw_r'])
#     d['M_200c'] = convert_mass(halos['profile_nfw_M'])
#     d['c_200c'] = halos['profile_nfw_c']
#     Dc, lat, lon = cartesian_to_spherical(d['x'].values, d['y'].values, d['z'].values)
#     lat, lon = np.array(lat), np.array(lon)
#     d['theta'] = np.pi / 2. - lat # needed by astropaint in rad
#     d['phi'] = lon # needed by astropaint in rad
#     d['lon'], d['lat'] = np.rad2deg([lon, lat]) # store lon lat in deg
#     d['D_c'] = np.array(Dc)
#     d['redshift'] = transform.D_c_Mpc_to_redshift_interp(np.array(d['D_c']))
#     d['D_a'] = transform.D_c_to_D_a(d['D_c'], d['redshift']) # angular diameter distance
#     d['R_200c_phys'] = transform.D_c_to_D_a(d['R_200c'], d['redshift'])  
#     d['R_s'] = np.true_divide(d['R_200c'], d['c_200c']) # needed by astropaint in Mpc phys
#     d['R_ang_200c'] = transform.radius_to_angsize(d['R_200c_phys'], d['D_a'], arcmin=True) # needed by astropaint in arcmin
#     d['rho_s'] = transform.M_200c_to_rho_s(d['M_200c'], d['redshift'], d['R_200c'], d['c_200c']) # needed by astropaint in mass/vol (physical) 

#     # use rho_s as 1+delta column
#     rho_m = cosmo.critical_density(z=d['redshift'])*cosmo.Om(z=d['redshift'])
#     rho_m = rho_m.to(u.M_sun/u.Mpc**3)
#     d['rho_s'] = np.array(rho_m.value)

#     import pudb; pudb.set_trace();
#     pass

#     J_cart2sph = transform.get_cart2sph_jacobian(d['theta'].values, d['phi'].values)
#     v_cart = np.array([d['v_x'], d['v_y'], d['v_z']])
#     d['v_r'], d['v_th'], d['v_ph'] = np.einsum('ij...,i...->j...', J_cart2sph, v_cart)
#     d['v_lat'] = -d['v_th']
#     d['v_lon'] = d['v_ph'] 

#     return Catalog(d)

    
    # catalog = Catalog(d, calculate_redshifts=True)


    # Dc, lat, lon = cartesian_to_spherical(d['x'].values, d['y'].values, d['z'].values)
    # d['D_c'], d['lat'], d['lon'] = np.array(Dc), np.array(lat), np.array(lon)
    # d['redshift'] = transform.D_c_Mpc_to_redshift_interp(np.array(d['D_c']))
    # d['theta'] = np.pi / 2. - d['lat']
    # d['phi'] = d['lon']
    # d['lon'], d['lat'] = np.rad2deg((d['lon'], d['lat']))
    # d['D_a'] = transform.D_c_to_D_a(d['D_c'], d['redshift']) # angular diameter distance
    # d['R_200c'] = transform.D_c_to_D_a(d['rvir'], d['redshift']) # Mpc physical
    # d['M_200c'] = d['Mvir']
    # d['c_200c'] = transform.ConcentrationDuffy08(mass_def='200c')._concentration(d['M_200c'], a=1/(1+d['redshift']))
    # d['R_s'] = np.true_divide(d['R_200c'], d['c_200c'])
    # d['R_ang_200c'] = transform.radius_to_angsize(d['R_200c'], d['D_a'], arcmin=True)
    # d['rho_s'] = transform.M_200c_to_rho_s(d['M_200c'], d['redshift'], d['R_200c'], d['c_200c'])


# def get_painted_shell(catalog, nside, template, nside_hires=4096, R_times=5):

#     from astropaint import Canvas, Painter

#     # prepare objects
#     canvas = Canvas(catalog, nside=nside_hires, R_times=R_times)
#     painter = Painter(template=template)

#     # main magic
#     time_start = time.time()
#     painter.spray(canvas)
#     time_elapsed = time.time() - time_start
#     LOGGER.info(f'painting time {time_elapsed/60.:4.1f} min, painting speed {time_elapsed/len(catalog.data):1.4f}s per halo')

#     m_hires = np.array(canvas.pixels)

#     LOGGER.info(f'changing resolution {nside_hires} -> {nside}')
#     m = hp.ud_grade(m_hires, nside_out=nside, power=-2)

#     return m



    # create the halos
    # baryon_shells, shell_stats_arr, shells_back = baryonification.get_baryon_shells(params_bary, 
    #                                                                                 shells=particle_shells,
    #                                                                                 shells_info=particle_shells_info,
    #                                                                                 halos=halos,
    #                                                                                 delta_shell=False,
    #                                                                                 nside_out=int(conf['baryonification']['nside_out']),
    #                                                                                 test=test,
    #                                                                                 using_mc_relation=conf['baryonification']['tag']=='v11')



# def get_astropaint_catalog(halos, tag):

#     if tag == 'v1':
#         return get_astropaint_catalog_v1(halos)
#     elif tag == 'v11':
#         return get_astropaint_catalog_v11(halos)

# class Catalog():

#     def __init__(self, data):

#         self.size = len(data)
#         self.data = data

# def get_astropaint_catalog_v11(halos):

#     import pandas as pd
#     from astropy.cosmology import current_cosmo as cosmo
#     from astropaint import transform
#     from astropy.coordinates import cartesian_to_spherical

#     dtype = {"names": ["x", "y", "z", "v_x", "v_y", "v_z", "R_ang_200c", "R_s", "rho_s", "theta", "phi"], "formats": 11 * [np.float32]}
#     d =  pd.DataFrame(np.zeros(len(halos), dtype))

#     # convert to Mpc and M_sun units
#     convert_position = lambda c: c / 1000 / (cosmo.H0.value/100.) # position in kpc/h -> Mpc
#     convert_mass = lambda m: m / (cosmo.H0.value/100.) # mass M_sun/h -> M_sun
#     convert_radius = lambda r: r / (cosmo.H0.value/100.) # radius Mpc/h -> Mpc
#     for c in ['x', 'y', 'z']: d[c] = convert_position(halos[c])
#     d['R_200c'] = convert_radius(halos['r_200c'])
#     d['M_200c'] = convert_mass(halos['m_200c'])
#     d['c_200c'] = halos['c_200c']
#     Dc, lat, lon = cartesian_to_spherical(d['x'].values, d['y'].values, d['z'].values)
#     lat, lon = np.array(lat), np.array(lon)
#     d['theta'] = np.pi / 2. - lat # needed by astropaint in rad
#     d['phi'] = lon # needed by astropaint in rad
#     d['lon'], d['lat'] = np.rad2deg([lon, lat]) # store lon lat in deg
#     d['D_c'] = np.array(Dc)
#     d['redshift'] = transform.D_c_Mpc_to_redshift_interp(np.array(d['D_c']))
#     d['D_a'] = transform.D_c_to_D_a(d['D_c'], d['redshift']) # angular diameter distance
#     d['R_200c_phys'] = transform.D_c_to_D_a(d['R_200c'], d['redshift'])  
#     d['R_s'] = np.true_divide(d['R_200c'], d['c_200c']) # needed by astropaint in Mpc phys
#     d['R_ang_200c'] = transform.radius_to_angsize(d['R_200c_phys'], d['D_a'], arcmin=True) # needed by astropaint in arcmin
#     d['rho_s'] = transform.M_200c_to_rho_s(d['M_200c'], d['redshift'], d['R_200c'], d['c_200c']) # needed by astropaint in mass/vol (physical) 

#     # use rho_s as 1+delta column (physical)
#     rho_m = cosmo.critical_density(z=d['redshift'])*cosmo.Om(z=d['redshift'])
#     rho_m = rho_m.to(u.M_sun/u.Mpc**3)
#     d['rho_s'] = np.array(rho_m.value)

#     import pudb; pudb.set_trace();
#     pass

#     J_cart2sph = transform.get_cart2sph_jacobian(d['theta'].values, d['phi'].values)
#     v_cart = np.array([d['v_x'], d['v_y'], d['v_z']])
#     d['v_r'], d['v_th'], d['v_ph'] = np.einsum('ij...,i...->j...', J_cart2sph, v_cart)
#     d['v_lat'] = -d['v_th']
#     d['v_lon'] = d['v_ph'] 

#     return Catalog(d)



    # # adopt units
    # halos = mpc_to_gpc(halos, fields=['x', 'y', 'z', 'nfw_r'])
    # sorting = np.argsort(halos['shell_id'])
    # halos = halos[sorting]

    # # we split the halos by shell
    # split_indices = get_split_indices(halos["shell_id"])
    # shelled_halos = np.split(halos, indices_or_sections=split_indices)

    # shell_info = read_shell_data(filename_halo)

    # loop over shells
    # shells_stats, shells_back, baryon_shells = [], [], []
    # for i, h in LOGGER.progressbar(list(enumerate(shelled_halos)), desc='displacing shells', at_level='warning'):


            # s = shells_info[shells_info["shell_id"] == h[0]["shell_id"]][0]
            # shell_index = match_shell(shell_info=shells_info, halo_shell=s)


        # halos = convert_Mpc_to_Gpc(halos, fields=['nfw_r'])


    # orig_shell = hp.ud_grade(particle_shells.astype(np.uint32), nside_out=nside_out, power=-2)
    # orig_shell = particle_shells if nside_out == nside_in else np.array([hp.ud_grade(p.astype(np.uint32), nside_out=nside_out, power=-2) for p in particle_shells])
    # LOGGER.info(f'storing {filename_out} {diff_shell.nbytes/1e9:4.2f} GB')
    # LOGGER.info(f'compressing arrays:')
    # LOGGER.info('orig_shell={:4.2f} GB'.format(orig_shell.nbytes/1e9) )
    # LOGGER.info('diff_shell_inds={:4.2f} GB'.format(diff_shell_inds.nbytes/1e9) )
    # LOGGER.info('diff_shell_vals={:4.2f} GB'.format(diff_shell_vals.nbytes/1e9) )
    # LOGGER.info('shell_infos_arr={:4.2f} GB'.format(shell_infos_arr.nbytes/1e9) )


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

def show_help():

    help_text = """
    run_paramtables usage instructions
    ======================================

    Objective: create various parameter tables for cosmogrid processing

    The app is has the following commands:
    (1) help: display this help text
    (2) baryon_params: create a table with baryon parameters, which can overwrite the fiducial ones from CosmoGridV1, one parameter set per simulation
    (3) shell_permutations: create shell permutation table used for projecting raw lightcone into probe maps
        
    Individual commands:

    (1) run_paramtables.py baryon_params
    Input: CosmoGrid_metainfo.pkl.gz (in this repository)
    Output: CosmoGridV1_baryon_params_override.h5 (in args/dir_out)

    (2) run_paramtables shell_permutations
    Input: CosmoGrid_metainfo.pkl.gz (in this repository)
    Output: metainfo_perms.npy (in args/dir_out)


    """
    print(help_text)


def setup(args):

    description = 'Create various parameter tables'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('command', type=str, default='help', choices=('help', 'shell_permutations', 'baryon_params'), 
                        help='command to run, "help" for instructions')
    parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
                        help='logging level')
    parser.add_argument('--config', type=str, required=True, 
                        help='configuration yaml file')
    parser.add_argument('--dir_out', type=str, required=True, 
                        help='output dir for the results')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    args = parser.parse_args(args)

    utils_logging.set_all_loggers_level(args.verbosity)

    # get absolute paths
    args.config = utils_io.get_abs_path(args.config)
    args.dir_out = utils_io.get_abs_path(args.dir_out)


    return args

def main(indices, args):

    from UFalcon import probe_weights

    # make tf records

    args = setup(args)
    conf = utils_config.load_config(args.config)
    utils_io.robust_makedirs(args.dir_out)

    if args.command == 'help':

        show_help()
        sys.exit(0)
    
    elif args.command == 'baryon_params':

        barylist = create_baryonification_parameters(conf, args)    
        filepath_barylist = get_filepath_barylist(args.dir_out, conf['tag'])
        store_barylist(filepath_barylist, barylist)


    elif args.command == 'shell_permutations':

        permlist = create_shell_permutations(conf, args) 
        filepath_permlist = get_filepath_permlist(args.dir_out)
        store_permlist(filepath_permlist, permlist)

    yield 0

def resources(args):

    if type(args) is list:
        args = setup(args)
        
    res = {'main_memory': 4000,
           'main_time_per_index': 1, # hours
           'main_scratch':4000}

    return res


def store_barylist(fpath, metainfo):
    
    LOGGER.info(f'storing {fpath}')

    with h5py.File(fpath, 'w') as f:

        for k, v in metainfo.items():

            if k != 'shell_info':

                k_ = k.rstrip('/')
                v_ = utils_arrays.unicode_to_ascii(v)

                f.create_dataset(name=k_, data=v_, compression='lzf')
                LOGGER.info(f[k_])

            else:

                is_ = 0
                for ks,vs in metainfo['shell_info'].items():
                    ns = f'shell_info/{ks}'.rstrip('/')
                    f.create_dataset(name=ns, data=vs, compression='lzf')
                    is_ += 1
                LOGGER.info(f'stored shell_info with {is_} datasets')

    LOGGER.info(f'stored {fpath}')
    LOGGER.info(f'---> make sure to use this table in subsequent baryonification processing and shell projection, this filepath should be added to the config under paths/metainfo_bary')
    LOGGER.info(f'---> make sure to use the total number of grid sims to baryonify in pipeline tasks {len(metainfo["simulations/grid"])}')
    LOGGER.info(f'---> make sure to use the total number of fiducials sims to baryonify in pipeline tasks {len(metainfo["simulations/fiducial"])}')
    LOGGER.info(f'---> make sure to use the total number of benchmark sims to baryonify in pipeline tasks {len(metainfo["simulations/benchmark"])}')
    LOGGER.info(f'---> make sure to use the total number of all sims to baryonify in pipeline tasks {len(metainfo["simulations/all"])}')

def create_baryonification_parameters(conf, args):
    

    # add bary params to lists
    conf_bary = conf['baryonification']['new_bary_params']
    bary_params = list(conf_bary.keys())
    bary_variable = [p for p in bary_params if 'fixed' not in conf_bary[p].keys()]

    # get all simulation list
    simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_simulations_list(set_type='all')
        

    ###################################
    #
    # grid
    #
    ###################################

    LOGGER.info('======================> creating specifications for baryonified grid sims')

    simslist_grid, parslist_grid, _ = utils_cosmogrid.get_simulations_list(set_type='grid')
    simslist_grid = utils_arrays.add_cols(simslist_grid, names=[f'bary_{p}' for p in bary_params])
    parslist_grid = utils_arrays.add_cols(parslist_grid, names=[f'bary_{p}' for p in bary_params])

    # create sobol sequence - extend the existing one
    import sobol
    n_pkd_dims = 5
    n_bary_dims = len(bary_variable)
    samp = sobol.sample(dimension=n_pkd_dims+n_bary_dims, n_points=parslist_grid['sobol_index'].max())

    scale_seq = lambda x, lims: x*(lims[1]-lims[0]) + lims[0]
    rename_root = lambda l: [l_.replace('CosmoGrid/raw/', 'CosmoGrid/bary/') for l_ in l]

    i = 5
    for par in bary_params:

        LOGGER.info(f'======> grid {par}')

        if par not in bary_variable: 

            vals = conf_bary[par]['fixed']
            simslist_grid[f'bary_{par}'] = vals
            parslist_grid[f'bary_{par}'] = vals

        else:

            vals = scale_seq(samp[:,i], conf_bary[par]['grid'])
            parslist_grid[f'bary_{par}'] = vals[parslist_grid['sobol_index']-1]
            simslist_grid[f'bary_{par}'] = vals[simslist_grid['sobol_index']-1]
            i += 1

        pv = simslist_grid[f'bary_{par}']
        LOGGER.info(f'grid: param {par} {conf_bary[par]} min {np.min(pv):2.4e} max {np.max(pv):2.4e}' )


    parslist_grid['path_par'] = rename_root(parslist_grid['path_par'])
    simslist_grid['path_par'] = rename_root(simslist_grid['path_par'])
    simslist_grid['path_sim'] = rename_root(simslist_grid['path_sim'])


    ###################################
    #
    # fiducial
    #
    ###################################

    LOGGER.info('======================> creating specifications for baryonified fiducial sims')

    simslist_fiducial, parslist_fiducial, _ = utils_cosmogrid.get_simulations_list(set_type='fiducial')
    simslist_fiducial = utils_arrays.add_cols(simslist_fiducial, names=[f'bary_{p}' for p in bary_params])
    parslist_fiducial = utils_arrays.add_cols(parslist_fiducial, names=[f'bary_{p}' for p in bary_params])

    # rename paths for existing sims
    simslist_fiducial['path_sim'] = rename_root(simslist_fiducial['path_sim'])
    simslist_fiducial['path_par'] = rename_root(simslist_fiducial['path_par'])
    parslist_fiducial['path_par'] = rename_root(parslist_fiducial['path_par'])

    # add deltas

    def add_delta(par, val, val_delta, id_par, id_sim, direction):

        # update param table

        param_row = pars_row_centr.copy()
        param_row['delta'] = f'delta_bary_{par}_{direction}'
        param_row[f'bary_{par}'] = val + val_delta # val_m should be negative
        param_row['id_param'] = id_par
        param_row['path_par'] = param_row['path_par'][0].replace('cosmo_fiducial', f'cosmo_delta_bary_{par}_{direction}')
            
        # update sim table

        # set columns
        sims_rows = sims_rows_centr.copy()
        sims_rows['delta'] = f'delta_bary_{par}_{direction}'
        sims_rows[f'bary_{par}'] = val + val_delta # val_m should be negative
        sims_rows['id_param'] = id_sim + np.arange(len(sims_rows))
        sims_rows['path_par'] = sims_rows['path_par'][0].replace('cosmo_fiducial', f'cosmo_delta_bary_{par}_{direction}')
        sims_rows['path_sim'] = sims_rows['path_sim'][0].replace('cosmo_fiducial', f'cosmo_delta_bary_{par}_{direction}')

        # rename paths to include bary key
        sims_rows['path_par'] = rename_root(sims_rows['path_par'])
        sims_rows['path_sim'] = rename_root(sims_rows['path_sim'])
        param_row['path_par'] = rename_root(param_row['path_par'])

        # expand directory names
        for i in range(len(sims_rows['path_sim'])):
            sims_rows['path_sim'][i] = sims_rows['path_sim'][i].replace('run_0000', f'run_{i:04d}')

        return param_row, sims_rows

    list_pars_new_deltas = []
    list_sims_new_deltas = []

    for par in bary_params:

        LOGGER.info(f'======> fiducial central {par}')

        vals = conf_bary[par]['fiducial'][0] if par in bary_variable else conf_bary[par]['fixed']
        simslist_fiducial[f'bary_{par}'] = vals
        parslist_fiducial[f'bary_{par}'] = vals

        pv = simslist_fiducial[f'bary_{par}']
        LOGGER.info(f'fiducial: param {par} {conf_bary[par]} min {np.min(pv):2.4e} max {np.min(pv):2.4e}' )

    pars_row_centr = parslist_fiducial[[0]].copy() # copy fiducial row
    select = simslist_fiducial['path_par']=='CosmoGrid/bary/fiducial/cosmo_fiducial'
    sims_rows_centr = simslist_fiducial[select].copy()

    id_par = np.max(parslist_fiducial['id_param'])+1 # initialize to existing max to continue adding parameters
    id_sim = np.max(simslist_fiducial['id_sim'])+1 # initialize to existing max to continue adding parameters

    for par in bary_params:

        LOGGER.info(f'======> fiducial {par}')

        if par in bary_variable: 

            val, val_m, val_p = conf_bary[par]['fiducial']

            # set central value
            simslist_fiducial[f'bary_{par}'] = val
            parslist_fiducial[f'bary_{par}'] = val
            pv = simslist_fiducial[f'bary_{par}']
            LOGGER.info(f'fiducial: param {par} {conf_bary[par]} min {np.min(pv):2.4e} max {np.max(pv):2.4e}' )

            # add deltas minus
            param_row, sims_rows = add_delta(par, val, val_m, id_par, id_sim, direction='m')
            list_pars_new_deltas.append(param_row)
            list_sims_new_deltas.append(sims_rows)
            pv = sims_rows[f'bary_{par}']
            shell_info_all[param_row[0]['path_par']] = shell_info_all['CosmoGrid/raw/fiducial/cosmo_fiducial']
            LOGGER.info(f'fiducial: param -δ {par} {conf_bary[par]} min {np.min(pv):2.4e} max {np.max(pv):2.4e}' )
            id_par += 1
            id_sim += len(sims_rows_centr)

            # add deltas plus
            param_row, sims_rows = add_delta(par, val, val_p, id_par, id_sim, direction='p')
            list_pars_new_deltas.append(param_row)
            list_sims_new_deltas.append(sims_rows)
            pv = sims_rows[f'bary_{par}']
            shell_info_all[param_row[0]['path_par']] = shell_info_all['CosmoGrid/raw/fiducial/cosmo_fiducial']
            LOGGER.info(f'fiducial: param +δ {par} {conf_bary[par]} min {np.min(pv):2.4e} max {np.max(pv):2.4e}' )
            id_par += 1
            id_sim += len(sims_rows_centr)

    pars_new_deltas = np.concatenate(list_pars_new_deltas)
    sims_new_deltas = np.concatenate(list_sims_new_deltas)

    LOGGER.info(f'fiducial: new deltas pars table rows {len(pars_new_deltas)}' )
    LOGGER.info(f'fiducial: new deltas sims table rows {len(sims_new_deltas)}' )

    parslist_fiducial = np.concatenate([parslist_fiducial, pars_new_deltas])
    simslist_fiducial = np.concatenate([simslist_fiducial, sims_new_deltas])

    LOGGER.info(f'fiducial: new pars table {len(parslist_fiducial)}' )
    LOGGER.info(f'fiducial: new sims table {len(simslist_fiducial)}' )


    ###################################
    #
    # benchmarks
    #
    ###################################

    # load benchmark
    simslist_benchmark, parslist_benchmark, _ = utils_cosmogrid.get_simulations_list(set_type='benchmark')
    simslist_benchmark = utils_arrays.add_cols(simslist_benchmark, names=[f'bary_{p}' for p in bary_params])
    parslist_benchmark = utils_arrays.add_cols(parslist_benchmark, names=[f'bary_{p}' for p in bary_params])

    # rename paths for existing sims
    simslist_benchmark['path_sim'] = rename_root(simslist_benchmark['path_sim'])
    parslist_benchmark['path_par'] = rename_root(parslist_benchmark['path_par'])

    for par in bary_params:

        LOGGER.info(f'======> benchmark {par}')

        if par not in bary_variable: 

            vals = conf_bary[par]['fixed']
            simslist_benchmark[f'bary_{par}'] = vals
            parslist_benchmark[f'bary_{par}'] = vals

            pv = simslist_benchmark[f'bary_{par}']
            LOGGER.info(f'fiducial: param {par} {conf_bary[par]} min {np.min(pv):2.4e} max {np.min(pv):2.4e}' )

        else:

            val, val_m, val_p = conf_bary[par]['fiducial']

            # set central value
            simslist_benchmark[f'bary_{par}'] = val
            parslist_benchmark[f'bary_{par}'] = val
            pv = simslist_benchmark[f'bary_{par}']
            LOGGER.info(f'fiducial: param {par} {conf_bary[par]} min {np.min(pv):2.4e} max {np.max(pv):2.4e}' )




    # post-process some parameters if needed

    pars_log10 = ['Mc']

    for par in pars_log10:
        
        parslist_grid[f'bary_{par}']      = 10**(parslist_grid[f'bary_{par}'])
        simslist_grid[f'bary_{par}']      = 10**(simslist_grid[f'bary_{par}'])
        parslist_fiducial[f'bary_{par}']  = 10**(parslist_fiducial[f'bary_{par}'])
        simslist_fiducial[f'bary_{par}']  = 10**(simslist_fiducial[f'bary_{par}'])
        parslist_benchmark[f'bary_{par}'] = 10**(parslist_benchmark[f'bary_{par}'])
        simslist_benchmark[f'bary_{par}'] = 10**(simslist_benchmark[f'bary_{par}'])

    # create "all" tables
    
    parslist_all = np.concatenate([parslist_fiducial, parslist_grid, parslist_benchmark])
    simslist_all = np.concatenate([simslist_fiducial, simslist_grid, simslist_benchmark])
    LOGGER.info(f'created merged table parslist_all {len(parslist_all)}')
    LOGGER.info(f'created merged table simslist_all {len(simslist_all)}')

    # final check

    def plot_stats(arr, par, tag):
        v = arr[f'bary_{par}']
        LOGGER.info(f'{tag:<40s} min={np.min(v):2.4e} max={np.max(v):2.4e} nun_unique={len(np.unique(v))} conf={conf_bary[par]}')


    for par in bary_params:

        LOGGER.info(f'=======> check param {par}')
        plot_stats(parslist_grid, par, tag=f'{par} parslist_grid')
        plot_stats(simslist_grid, par, tag=f'{par} simslist_grid')
        plot_stats(parslist_fiducial, par, tag=f'{par} parslist_fiducial')
        plot_stats(simslist_fiducial, par, tag=f'{par} simslist_fiducial')
        plot_stats(parslist_benchmark, par, tag=f'{par} parslist_benchmark')
        plot_stats(simslist_benchmark, par, tag=f'{par} simslist_benchmark')
        plot_stats(parslist_all, par, tag=f'{par} parslist_all')
        plot_stats(simslist_all, par, tag=f'{par} simslist_all')


    # create new shell_info_all

    shell_info_bary_all = {}
    for k,v in shell_info_all.items():
        kv = k.replace('CosmoGrid/raw/', 'CosmoGrid/bary/')
        shell_info_bary_all[kv] = utils_arrays.rewrite(v)

    metainfo = {'parameters/grid' : parslist_grid,
                'simulations/grid' : simslist_grid,
                'parameters/fiducial' : parslist_fiducial,
                'simulations/fiducial' : simslist_fiducial,
                'parameters/benchmark' : parslist_benchmark,
                'simulations/benchmark' : simslist_benchmark,
                'parameters/all' : parslist_all,
                'simulations/all' : simslist_all,
                'shell_info': shell_info_bary_all }


    return metainfo



def create_shell_permutations(conf, args):

    # get simulation list
    simslist_all, parslist_all, shellinfo_all = utils_cosmogrid.get_baryonified_simulations_list(conf, set_type='all')
    LOGGER.info(f'loaded {len(parslist_all)} parameters and {len(simslist_all)} simulations')
    parslist_all = utils_arrays.add_cols(parslist_all, names=['row_id_parslist:i4'], data=np.arange(len(parslist_all)))


    # get redshift bins
    nz_info = conf['redshifts_nz']
    n_redshift_bins = len(nz_info)
    LOGGER.info(f'using {n_redshift_bins} redshift bins')
    for i, p in enumerate(nz_info):
        LOGGER.info(f'bin {i+1}/{n_redshift_bins} {p}')

    # create table with
    # parslist_id
    # dirname_out
    # redshift_bin_name delta_meanz
    # redshift_bin_name delta_sigmaz
    # parslist entries
    # permutation index
    # halton sequence id


    new_dtype = [('dirname_perm', 'a512'),
                 ('id_redshift_perturb', 'i4'),
                 ('id_perm', 'i4')]
    
    for i, p in enumerate(nz_info):

        if p['name'] != 'cmb':
    
            new_dtype += [(p['name'] + '__delta_meanz', 'f4'),
                          (p['name'] + '__delta_sigmaz', 'f4')]

    dtype = np.dtype(parslist_all.dtype.descr + new_dtype)

    sim_types = ['fiducial', 'grid', 'benchmarks']



    # default number of permutations per sim
    n_perms_per_type = {'fiducial': 200, 'grid': 7, 'benchmarks': 7} 

    # set from config
    if 'n_perms_per_cosmo' in conf['projection'].keys():
        n_perms_per_type.update(conf['projection']['n_perms_per_cosmo'])

    # if to use same perturbation sequence for simulations in these categories
    par_fixed = {'fiducial': True, 'grid': False, 'benchmarks': True}

    LOGGER.info(f'number of shell permutations: {n_perms_per_type}')


    def get_permlist_for_simulation_type(sim_type):
    
        n_perm = n_perms_per_type[sim_type]
        select = np.array([sim_type+'/' in p for p in parslist_all['path_par']])
        parslist = parslist_all[select]
        n_par = np.count_nonzero(select)
        n_total = n_perm * n_par

        permlist = np.empty(n_total, dtype)
        LOGGER.info(f'creating permutation table for sim_type={sim_type} n_perm={n_perm} resetting sequence for each parameter {par_fixed[sim_type]} with n_rows={len(permlist)}')

        h = 0
        n = 0

        if not par_fixed[sim_type]:
            samples_z_perturb = get_joint_param_grid(nz_info, len_seq=n_total, sequence='latin_hypercube')

        # loop over cosmology parameters
        for i in range(n_par):

            params_current = parslist[i]

            # loop over permutations
            for j in range(n_perm):

                if par_fixed[sim_type]:

                    permlist = add_central_redshift_perturb_value(nz_info, permlist, n)
                
                else:
                    permlist = add_scatter_perturb_value(samples_z_perturb, permlist, h, n)

                # fill parameters
                for c in parslist.dtype.names:
                    permlist[n][c] = parslist[i][c]

                # info for the permutation
                dirname_out = get_dirname_permuted_maps(dir_out=args.dir_out,
                                                        project_tag=conf['tag'],
                                                        cosmo_params=params_current,
                                                        id_perm=j)

                permlist['dirname_perm'][n] = dirname_out.replace(args.dir_out, '').lstrip('/')
                permlist['id_redshift_perturb'][n] = h
                permlist['id_perm'][n] = j

                h+=1
                n+=1

        return permlist

    permlists = []
    for sim_type in sim_types:
        permlist = get_permlist_for_simulation_type(sim_type)
        permlists.append(permlist)
    permlist = np.concatenate(permlists)

    return permlist

def add_central_redshift_perturb_value(nz_info, permlist, n):

    for i, p in enumerate(nz_info):

        if p['name'] != 'cmb':
    
            par = p['name'] + '__delta_meanz'
            if 'delta_mean_z' in p.keys():
                permlist[n][par] = p['delta_mean_z'][0]

            par = p['name'] + '__delta_sigmaz'
            if 'delta_sigma_z' in p.keys():
                permlist[n][par] = p['delta_sigma_z'][0]
            
    return permlist

def add_scatter_perturb_value(samples_z_perturb, permlist, h, n):
                
    for c in samples_z_perturb.dtype.names:
        permlist[n][c] = samples_z_perturb[h][c]

    return permlist


def get_joint_param_grid(nz_info, len_seq = 100000, sequence='halton'):

    n_redshift_bins = len(nz_info)

    collect_var = []
    labels_var = []
    labels_extra = []
    import chaospy
    for z in range(n_redshift_bins):

        if 'cmb' in nz_info[z]['name']:
            continue

        label = nz_info[z]['name'] + '__delta_meanz'
        if 'delta_mean_z'  in nz_info[z]:
            mu, sig = nz_info[z]['delta_mean_z']
            dist = chaospy.TruncNormal(mu=mu, sigma=sig, lower=mu-3*sig, upper=mu+3*sig)
            labels_var.append(label)
            collect_var.append(dist)
        
        else:
            labels_extra.append(label)

        label = nz_info[z]['name'] + '__delta_sigmaz'
        if 'delta_sigma_z'  in nz_info[z]:
            mu, sig = nz_info[z]['delta_sigma_z']
            dist = chaospy.TruncNormal(mu=mu, sigma=sig, lower=mu-3*sig, upper=mu+3*sig)
            labels_var.append(label)
            collect_var.append(dist)
        else:
            labels_extra.append(label)

    LOGGER.info(f'collected {len(collect_var)} random variables, creating joint distribution with sequence={sequence}')


    no_perturb = 0. # 0 is the fiducial value for perturbations=no perturbation

    np.random.seed(101212)

    if len(collect_var) > 0:
        # sample new points
        joint_var = chaospy.J(*collect_var)
        samples_z_perturb = utils_arrays.arr_to_rec(arr=joint_var.sample(len_seq, rule=sequence).T, 
                                                    cols=labels_var)    
        # add the fields that were not sampled
        samples_z_perturb = utils_arrays.add_cols(samples_z_perturb, names=labels_extra, data=no_perturb) 

    else:

        samples_z_perturb = utils_arrays.arr_to_rec(arr=np.zeros((len_seq, 2)), 
                                                    cols=labels_var)    



    samples_z_perturb = utils_arrays.add_cols(samples_z_perturb, names=labels_extra, data=no_perturb) 

    return samples_z_perturb


def store_permlist(fname, permlist):

    np.save(fname, permlist)
    LOGGER.info(f'stored {fname} with {len(permlist)} rows')
    LOGGER.warning(f'---> now update config file paths/redshift_perturbations_list with {fname}')
    LOGGER.warning(f'---> now update pipeline command with number of tasks --tasks=0>{len(permlist)}')

if __name__ == '__main__':

    next(main([0], sys.argv[1:]))
    




# code graveyard

    # # index calculations
    # params_current = parslist_use[index_par] # this doesn't use anything about the sim, just the params, this is here to match the indexing scheme with the "no permutation" version
    # path_params_current = params_current['path_par']
    # shell_info_current = shell_info_all[path_params_current]
    # n_sims_avail = np.count_nonzero(simslist_all['path_par']==path_params_current)
    # n_sims_use = min(float(conf['projection']['n_max_sims_use']), n_sims_avail)
    # shell_groups = utils_shell_permutation.get_shell_groups(shell_info_current, n_max_replicas, Lbox=params_current['box_size_Mpc_over_h'])

    # n_shell_groups = len(shell_groups)

    # seed = conf['projection']['shell_perms_seed']+index_perm

    # np.random.seed(seed)

    # LOGGER.info(f'using random seed={seed} for shell permutations')

    # list_perms_info = []

    #     perms_info = utils_shell_permutation.get_shell_permutation_sequence(n_shell_groups, n_sims_use)


# def get_redshift_perturb(index, nz_info, offset=0):

#     from cosmogridv1.utils_redshift import sample_redshift_perturb_params

#     # first permutation always has 

#     if ('delta_mean_z' in nz_info.keys()) and ('delta_sigma_z' in nz_info.keys()): 
        
#         if index==-1:

#             redshift_params = [nz_info['delta_mean_z'][0], nz_info['delta_sigma_z'][0]]

#         else:

#             redshift_params = sample_redshift_perturb_params(i=index, 
#                                                              mu=[nz_info[c][0] for c in ['delta_mean_z','delta_sigma_z']], # 0-th is the mean 
#                                                              sigma=[nz_info[c][1] for c in ['delta_mean_z','delta_sigma_z']], # 1-st is the standard deviation
#                                                              offset=offset)[0]

#     elif ('delta_mean_z' in nz_info.keys()):

#         if index==-1:

#             redshift_params = [nz_info['delta_mean_z'][0], 0]

#         else:

#             redshift_delta = sample_redshift_perturb_params(i=index, 
#                                                             mu=[nz_info['delta_mean_z'][0]], # 0-th is the mean 
#                                                             sigma=[nz_info['delta_mean_z'][1]], # 1-st is the standard deviation
#                                                             offset=id_bin*bin_offset)
#             redshift_params = [redshift_delta[0,0], 1]


#     elif ('delta_sigma_z' in nz_info.keys()):

#         if index==-1:

#             redshift_params = [nz_info['delta_mean_z'][0], 0]

#         else:

#             redshift_delta = sample_redshift_perturb_params(i=index, 
#                                                             mu=[nz_info['delta_sigma_z'][0]], # 0-th is the mean 
#                                                             sigma=[nz_info['delta_sigma_z'][1]], # 1-st is the standard deviation
#                                                             offset=id_bin*bin_offset)
#             redshift_params = [0, redshift_delta[0,0]]

#     else:

#         redshift_params = [0, 0]

#     LOGGER.debug(f'index={index} using redshift perturbation delta_mean_z={redshift_params[0]} delta_sigma_z={redshift_params[1]}')

#     return redshift_params



# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2022
author: Tomasz Kacprzak
"""

import os, sys, warnings, argparse, h5py, numpy as np, time, logging, itertools, shutil
from cosmogridv11 import utils_io, utils_logging, utils_config, utils_cosmogrid, utils_maps, utils_arrays
from cosmogridv11.filenames import *
import healpy as hp

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)


def get_redshift_perturb(index, nz_info, offset=0):

    from cosmogridv11.utils_redshift import sample_redshift_perturb_params

    # first permutation always has 

    if ('delta_mean_z' in nz_info.keys()) and ('delta_sigma_z' in nz_info.keys()): 
        
        if index==-1:

            redshift_params = [nz_info['delta_mean_z'][0], nz_info['delta_sigma_z'][0]]

        else:

            redshift_params = sample_redshift_perturb_params(i=index, 
                                                             mu=[nz_info[c][0] for c in ['delta_mean_z','delta_sigma_z']], # 0-th is the mean 
                                                             sigma=[nz_info[c][1] for c in ['delta_mean_z','delta_sigma_z']], # 1-st is the standard deviation
                                                             offset=offset)[0]

    elif ('delta_mean_z' in nz_info.keys()):

        if index==-1:

            redshift_params = [nz_info['delta_mean_z'][0], 0]

        else:

            redshift_delta = sample_redshift_perturb_params(i=index, 
                                                            mu=[nz_info['delta_mean_z'][0]], # 0-th is the mean 
                                                            sigma=[nz_info['delta_mean_z'][1]], # 1-st is the standard deviation
                                                            offset=id_bin*bin_offset)
            redshift_params = [redshift_delta[0,0], 0]


    elif ('delta_sigma_z' in nz_info.keys()):

        if index==-1:

            redshift_params = [nz_info['delta_mean_z'][0], 0]

        else:

            redshift_delta = sample_redshift_perturb_params(i=index, 
                                                            mu=[nz_info['delta_sigma_z'][0]], # 0-th is the mean 
                                                            sigma=[nz_info['delta_sigma_z'][1]], # 1-st is the standard deviation
                                                            offset=id_bin*bin_offset)
            redshift_params = [0, redshift_delta[0,0]]

    else:

        redshift_params = [0, 0]

    LOGGER.debug(f'index={index} using redshift perturbation delta_mean_z={redshift_params[0]} delta_sigma_z={redshift_params[1]}')

    return redshift_params



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


def setup(args):

    description = 'Make maps'
    parser = argparse.ArgumentParser(description=description, add_help=True)
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

def main(args):


    # make tf records

    args = setup(args)
    conf = utils_config.load_config(args.config)
    
    # get simulation list
    simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_simulations_list(set_type='all')
    parslist_all = utils_arrays.add_cols(parslist_all, names=['row_id_parslist:i4'], data=np.arange(len(parslist_all)))

    # get redshift bins
    nz_info = conf['redshifts_nz']
    n_redshift_bins = len(nz_info)
    LOGGER.info(f'using {n_redshift_bins} redshift bins')
    for i, p in enumerate(nz_info):
        LOGGER.info(f'bin {i+1}/{n_redshift_bins} {p}')

    samples_z_perturb = get_joint_param_grid(nz_info, len_seq = 100000, sequence='halton')

    # create table with
    # parslist_id
    # dirname_out
    # redshift_bin_name delta_meanz
    # redshift_bin_name delta_sigmaz
    # parslist entries
    # permutation index
    # halton sequence id


    new_dtype = [('dirname_perm', 'a512'),
                 ('halton_point_id', 'i4'),
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
    par_same_seq = {'fiducial': True, 'grid': False, 'benchmarks': True}

    LOGGER.info(f'number of shell permutations: {n_perms_per_type}')

    def get_permlist_for_simulation_type(sim_type):
    
        n_perm = n_perms_per_type[sim_type]
        select = np.array([sim_type+'/' in p for p in parslist_all['path_par']])
        parslist = parslist_all[select]
        n_par = np.count_nonzero(select)
        n_total = n_perm * n_par

        permlist = np.empty(n_total, dtype)
        LOGGER.info(f'creating permutation table for sim_type={sim_type} n_perm={n_perm} resetting sequence for each parameter {par_same_seq[sim_type]} with n_rows={len(permlist)}')

        h = 0
        n = 0
        for i in range(n_par):

            params_current = parslist[i]

            if par_same_seq[sim_type]:
                h=0

            for j in range(n_perm):

                if j == 0:
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
                permlist['halton_point_id'][n] = h
                permlist['id_perm'][n] = j

                if j>0:
                    h+=1
                
                n+=1

        return permlist

    permlists = []
    for sim_type in sim_types:
        permlist = get_permlist_for_simulation_type(sim_type)
        permlists.append(permlist)
    permlist = np.concatenate(permlists)

    utils_io.robust_makedirs(args.dir_out)
    filepath_permlist = get_filepath_permlist(args.dir_out)
    store_permlist(filepath_permlist, permlist)

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



def store_permlist(fname, permlist):

    np.save(fname, permlist)
    LOGGER.info(f'stored {fname} with {len(permlist)} rows')

if __name__ == '__main__':

    main(sys.argv[1:])



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

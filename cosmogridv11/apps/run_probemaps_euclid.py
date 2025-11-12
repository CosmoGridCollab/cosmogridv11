# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2022
author: Tomasz Kacprzak
"""

import os, warnings, argparse, h5py, numpy as np, time
from cosmogridv11 import utils_io, utils_logging, utils_config, utils_cosmogrid, utils_shells, utils_maps, utils_projection
from cosmogridv11.filenames import *
import healpy as hp
from cosmogridv11.copy_guardian import NoFileException
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

# https://github.com/mgatti29/LSS_forward_model/blob/main/Examples/Cosmogrid.ipynb
import LSS_forward_model
from LSS_forward_model.cosmology import *
from LSS_forward_model.lensing import *
from LSS_forward_model.maps import *
from LSS_forward_model.halos import *
from LSS_forward_model.tsz import *
from LSS_forward_model.theory import *


# define args
z_lim_low = 0 # for lss probes
z_lim_up = 3 # for lss probes
base_shift_nz = 0.0
kw_ufalcon = {'interpolation_kind': 'linear', 'z_lim_low': z_lim_low, 'z_lim_up': z_lim_up, 'shift_nz': base_shift_nz}


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
    parser.add_argument('--precopy', action='store_true',
                        help='Copy all sims before proceeding')

    args = parser.parse_args(args)

    utils_logging.set_all_loggers_level(args.verbosity)

    # get absolute paths
    args.config = utils_io.get_abs_path(args.config)
    if args.dir_out is not None:
        args.dir_out = utils_io.get_abs_path(args.dir_out)


    return args


def resources(args):

    if type(args) is list:
        args = setup(args)
    
    res = {'main_nsimult': 500,
           'main_memory':16000,
           'main_time_per_index':1, # hours
           'main_scratch':int(2000*args.num_maps_per_index),
           'merge_memory':64000,
           'merge_time':24,
           'pass': {'constraint': 'cpu', 'account': 'des', 'qos': 'shared'}
           } # perlmutter

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
            res['main_nsimult'] = 400
    
    return res


def main(indices, args):
    """
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

            files_out = project_single_sim(id_, args, conf)

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

    simslist_all, parslist_all, shellinfo_all = utils_cosmogrid.get_baryonified_simulations_list(conf, set_type='all')
    sim_current = simslist_all[index]
    shellinfo_current = shellinfo_all[sim_current['path_par']]

    LOGGER.info(f"=============================> index={index} sim={sim_current['path_par']}")

    # prepare output
    id_run = int(sim_current['path_sim'].split('/')[-1].split('run_')[1])
    dirpath_out = get_dirname_projected_maps(args.dir_out, sim_current, id_run=id_run, project_tag=conf['tag'])
    utils_io.robust_makedirs(dirpath_out)
    utils_io.ensure_permissions(dirpath_out, verb=True)

    files_variants = []
    for variant in conf['analysis_variants']:

        LOGGER.info(f'==============> maps for variant={variant}')
        
        euclid_map_projector(index,
                             dirpath_out,
                             variant,
                             nside_maps=1024, 
                             path_simulations=os.path.join(conf['paths']['cosmogrid_bary'].rstrip('CosmoGrid/bary/'), sim_current['path_sim']), 
                             path_meta=conf['paths']['metainfo_bary'], 
                             path_SC_corrections=os.path.join(LSS_forward_model.__path__[0], '../Data/SC_RR2_fit_nov6.npy'), 
                             path_nz_RR2=conf['paths']['redshifts_euclid'], 
                             path_data_cats=conf['paths']['data_cats_euclid'])


def euclid_map_projector(index, dirpath_out, variant, nside_maps, path_simulations, path_meta, path_SC_corrections, path_nz_RR2, path_data_cats, plots=False):
    """
    https://github.com/mgatti29/LSS_forward_model/blob/main/Examples/Cosmogrid.ipynb
    """


    filepath_out = get_filepath_projected_maps(dirpath_out, variant)

    # Make a Cosmogrid Euclid RR2 catalog/map


    # nuisance parameters ------------------------------
    nuisance_parameters = dict()
    nuisance_parameters['dz'] =  [0,0,0,0,0,0,0]
    nuisance_parameters['dm'] =  [1,1,1,1,1,1,1]
    nuisance_parameters['A_IA'] = 0.
    nuisance_parameters['eta_IA'] = 0.
    nuisance_parameters['bias_sc'] = [1,1,1,1,1,1,1]
    nuisance_parameters['rot'] = 0


    # Baryonification settings ------------------------------------------------
    # Ignore these for the moment - just use the baryonified shells from Tomasz!
    # baryons = {
    #     "enabled": False,
    #     "max_z_halo_catalog": 1.5,
    #     "mass_cut": 13,
    #     "do_tSZ": False,
    #     "base_params_path": "../Data/Baryonification_wl_tsz_flamingo_parameters.npy",
    #     "filename_new_params": "sys_baryo_0.npy",
    #     "values_to_update":  None, # or: {'Mc': 10**13,'theta_ej' : 4.} or draw_params_from_specs( {"M_c": (12.5, 15.5, "log10"),   "theta_ej": (3.0, 10.0, "lin"),    "eta": (-2.0, -0.1, "log10")} )
    # }

    SC_corrections = np.load(path_SC_corrections, allow_pickle=True).item()

    data_cats = np.load(path_data_cats,allow_pickle=True).item()

    nz_RR2 = np.load(path_nz_RR2,allow_pickle=True).item()  


    # Read cosmology & parameter setup

    sims_parameters = load_cosmogrid_siminfo(path_meta, int(index))
    cosmo_bundle = make_cosmo_bundle(sims_parameters)
    sims_parameters.update(nuisance_parameters)

    with h5py.File(path_meta,'r') as meta_info:
        z_low = np.array(meta_info[f'shell_info/{sims_parameters["path_par"]}'])['lower_z']
        z_high = np.array(meta_info[f'shell_info/{sims_parameters["path_par"]}'])['upper_z']
        d_low = np.array(meta_info[f'shell_info/{sims_parameters["path_par"]}'])['lower_com']
        d_high = np.array(meta_info[f'shell_info/{sims_parameters["path_par"]}'])['upper_com']

    z_edges = np.hstack([z_low[0],z_high])
    d_edges = np.hstack([d_low[0],d_high])

    # add some extra shells to redshift 6 --
    z_edges = np.hstack([z_edges,np.arange(z_high[-1],6.,0.12)[1:]])
    d_edges = np.hstack([z_edges,redshift_to_distance(np.arange(z_high[-1],6.,0.12)[1:],cosmo_bundle)*sims_parameters['h']])
    shells_info = make_shells_info_from_edges(z_edges, d_edges)


    # setup n(z)

    nz_RR2 = np.load(path_nz_RR2, allow_pickle=True).item()

    # -------------------------------------------------------
    nz_shifted, shells, steps, zeff_glass, ngal_glass = apply_nz_shifts_and_build_shells(
        z_rebinned=nz_RR2['z_rebinned'],
        nz_all=nz_RR2['nz_rebinned'],
        dz_values=sims_parameters["dz"],
        shells_info=shells_info,
    )

    # Plot -----------------------------------------------------------------

    if plots:
        LOGGER.info('plotting ngal cells')
        fig, axes = plt.subplots(1, 7, figsize=(20, 3), sharey=True)
        titles = ["Non-tomo"] + [f"Bin {i}" for i in range(1, 7)]
        for i, ax in enumerate(axes):
            ax.plot(zeff_glass, ngal_glass[i], label="data")
            ax.set_xlim(0, 6)
            ax.set_title(titles[i])
            if i == 0: ax.set_ylabel("Galaxy count")
            ax.set_xlabel("z_eff")
        axes[-1].legend(loc="upper right", fontsize="small")
        plt.tight_layout();
        filename_fig = filepath_out.replace('.h5', 'ngalbins.png')
        plt.savefig(filename_fig)
        plt.close()
        LOGGER.info(f'saved figure to {filename_fig}')



    # load density shells and compute shear field

    shells_type = 'nobaryon_shells' if 'dmo' in variant else 'baryonified_shells'
    particles = (utils_maps.load_v11_shells(os.path.join(path_simulations, 'baryonified_shells_v11.h5'), shells_type))['particles']
    density = particles / np.mean(particles, axis=1, keepdims=True) - 1

    #add high redshift shells --------------
    
    if len(shells)>len(density):
        LOGGER.info('applying high redshift shells')
        missing_shells = []
        for i in range(len(density),len(shells)):
            missing_shells.append(shells[i])
        density_to_be_added = add_shells(cosmo_bundle['pars_camb'],nside_maps = nside_maps,missing_shells = missing_shells)
    density = np.vstack([density,density_to_be_added])

    # shear field ---------------------------
    LOGGER.info('computing shear field')
    fields = compute_lensing_fields(density, shells, cosmo_bundle['pars_camb'], nside_maps, do_kappa=True, do_shear=True, do_IA=True)
    fields['density'] = density


    # Theory checks
    LOGGER.info('computing theory')
    theory = LimberTheory(cosmo_bundle['pars_camb'], lmax=4000, nonlinear="euclidemu")  # "euclidemu" | "mead" | "halofit"
    theory.set_Wshear(np.vstack([nz_RR2['z_rebinned'],nz_shifted]).T)
    Cgg = theory.cl_gg(nonlinear=True)

    LOGGER.info('integrating kappa')
    kappa_tomo = integrate_field(ngal_glass, fields["kappa"])

    LOGGER.info('computing Cls')
    Cls = np.array([(hp.anafast(kappa_tomo[tomo,:])) for tomo in range(len(ngal_glass))])



    # choose a safe ℓ-range common to all arrays
    
    ell_max = 2000
    pix = hp.pixwin(nside_maps)[:ell_max]
    ells = np.arange(ell_max)
    mask_ell = ells >= 2  # avoid ℓ=0,1
    if plots:
        LOGGER.info('plotting shear Cells')
        fig, axes = plt.subplots(1, 7, figsize=(20, 3), sharey=True)
        titles = ["Non-tomo"] + [f"Bin {i}" for i in range(1, 7)]
        for tomo, ax in enumerate(axes):
            ax.plot(Cls[tomo, :ell_max]/(Cgg[tomo, tomo, :ell_max] * (pix**2)))
            ax.plot(np.ones(ell_max),color = 'black')
            ax.plot(0.9*np.ones(ell_max),color = 'black')
            ax.plot(1.1*np.ones(ell_max),color = 'black')
            ax.set_ylim([0.5,2])
            ax.set_title(titles[tomo])

        ax.grid(True, which="both", ls=":", alpha=0.6)

        finame_fig = filepath_out.replace('.h5', 'shear_cells.png')
        plt.savefig(filename_fig)
        plt.close()
        LOGGER.info('saved figure to {filename_fig}')

    # Make mocks/maps
    cats_Euclid  = np.load(path_data_cats, allow_pickle=True).item()
    LOGGER.info('making WL sample')
    maps_WL, cat_WL = make_WL_sample(ngal_glass, zeff_glass, cosmo_bundle, sims_parameters, nside_maps, fields, cats_Euclid, 
                                                 SC_corrections=SC_corrections, 
                                                 do_catalog=False, 
                                                 include_SC=True)


    # store 
    store_products(filepath_out, maps_WL, cat_WL, kappa_tomo, Cls, Cgg, nuisance_parameters, ngal_glass)
    

    import pudb; pudb.set_trace();

def store_products(filepath_out, maps_WL, cat_WL, kappa_tomo, Cls, Cgg, nuisance_parameters, ngal_glass, cosmo_bundle):


    LOGGER.info('storing maps')
    import pudb; pudb.set_trace();

    # create indices set
    hp_indices = np.arange(hp.nside2npix(nside_maps))
    select = np.zeros(len(hp_indices), dtype=bool)
    for sample_name, sample_data in maps_WL.items():
        for field_name, field_data in sample_data.items():
            select |= (field_data != 0)
    hp_indices = hp_indices[select]
    LOGGER.info(f'selected {sum(select)}/{len(select)} pixels')
    


    with h5py.File(filepath_out, 'w') as f:

        LOGGER.info(f'storing hp_indices')
        f.create_dataset(name='hp_indices', data=hp_indices, shuffle=True, compression="gzip", compression_opts=4)

        LOGGER.info('storing maps')
        if maps_WL is not None:

            for sample_name, sample_data in maps_WL.items():
                for field_name, field_data in sample_data.items():
                    dset = f'maps_WL/{sample_name}/{field_name}'
                    LOGGER.info(f'storing {dset}')
                    f.create_dataset(name=dset, data=field_data[hp_indices], shuffle=True, compression="gzip", compression_opts=4)

        LOGGER.info('storing catalog')
        if cat_WL is not None:

            for sample_name, sample_data in cat_WL.items():
                for field_name, field_data in sample_data.items():
                    dset = f'cat_WL/{sample_name}/{field_name}'
                    LOGGER.info(f'storing {dset}')
                    f.create_dataset(name=dset, data=field_data[hp_indices], shuffle=True, compression="gzip", compression_opts=4)

        LOGGER.info('storing kappa_tomo')
        if kappa_tomo is not None:
            for i, kappa in enumerate(kappa_tomo):
                dset = f'kappa_tomo/{i}'
                LOGGER.info(f'storing {dset}')
                f.create_dataset(name=dset, data=kappa[hp_indices], shuffle=True, compression="gzip", compression_opts=4)
                
        if Cls is not None:
            LOGGER.info('storing Cls_sim')
            for i, Cl in enumerate(Cls):
                dset = f'Cls_sim/{i}'
                LOGGER.info(f'storing {dset}')
                f.create_dataset(name=dset, data=Cl, shuffle=True, compression="gzip", compression_opts=4)

        if Cgg is not None:
            LOGGER.info('storing Cls_theory')
            for i, Cl in enumerate(Cgg):
                dset = f'Cls_theory/{i}'
                LOGGER.info(f'storing {dset}')
                f.create_dataset(name=dset, data=Cl[i], shuffle=True, compression="gzip", compression_opts=4)

        if ngal_glass is not None:
            for i, ngal in enumerate(ngal_glass):
                dset = f'ngal_glass/{i}'
                LOGGER.info(f'storing {dset}')
                f.create_dataset(name=dset, data=ngal, shuffle=True, compression="gzip", compression_opts=4)


        # TODO: store nuisance parameters and cosmo bundle
        

    LOGGER.info(f'stored maps to {filepath_out}')
    



def missing(indices, args):

    args = setup(args)
    conf = utils_config.load_config(args.config)

    if conf['projection']['shell_perms'] == True:
        permlist_all = utils_cosmogrid.load_permutations_list(conf)
        simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_baryonified_simulations_list(conf, set_type='all')

    list_missing = []

    # loop over sims 

    for index in (pbar := LOGGER.progressbar(indices, desc='checking for missing results', at_level='info')):     

        if conf['projection']['shell_perms'] == False:

            raise Exception('not implemented yet')

        else:

            perm_ids = np.arange(index*args.num_maps_per_index, (index+1)*args.num_maps_per_index)
    
            for perm_id in perm_ids:

                perm_current = permlist_all[perm_id]
                index_perm =  perm_current['id_perm']
                index_par = perm_current['row_id_parslist']
                params_current = parslist_all[index_par] # this doesn't use anything about the sim, just the params, this is here to match the indexing scheme with the "no permutation" version
                dirname_out = get_dirname_permuted_maps(dir_out=args.dir_out,
                                                        project_tag=conf['tag'],
                                                        cosmo_params=params_current,
                                                        id_perm=index_perm)

                for variant in conf['analysis_variants']:
        
                    filepath_out = get_filepath_projected_maps(dirname_out, variant)

                    if not os.path.isfile(filepath_out):

                        LOGGER.debug(f'{perm_id: 6d} /{len(indices)*args.num_maps_per_index} file missing: {filepath_out}')
                        list_missing.append(index)
                        pbar.set_description_str(f"checking for missing results, found {len(list_missing)} missing")

                    else:

                        LOGGER.debug(f'{perm_id: 6d} /{len(indices)*args.num_maps_per_index} file OK:      {filepath_out}')



    list_missing = list(np.unique(list_missing))
    LOGGER.info(f'missing {len(list_missing)} indices:')
    LOGGER.info(','.join([str(i) for i in list_missing]))

    return list_missing


def get_filename_shells_for_variant(variant):

    # for CosmoGridV1.1
    if variant in ['v11dmb', 'v11dmo']:
        filename_shells = 'baryonified_shells_v11.h5'
        nside = 1024

    else:
        raise Exception(f'unknown analysis variant {variant}')

    return filename_shells, nside

def get_filename_haloshells_for_variant(variant):

    # for CosmoGridV1.1
    if variant in ['v11dmo']:
        filename_shells = 'haloshells_v11.h5'
        nside = 1024

    else:
        raise Exception(f'haloshells unavailable for analysis variant {variant}')

    return filename_shells, nside



def select_shell_group(shells, select):
    shells_select = {}
    for k, s in shells.items():
        shells_select[k] = s[select,:]
    return shells_select


def arr_row_str(a):

    s = ''
    for k in a.dtype.names:
        s += f'{k}={str(a[k]):>4s} '
    return s








# code graveyard

        # seed_highz = conf['projection']['highz_synfast_seed'] + sim_params['id_param']*10000 + id_perm
    # get useful parameters
     # this doesn't use anything about the sim, just the params, this is here to match the indexing scheme with the "no permutation" version
    # path_params_current = params_current['path_par']
    # shell_info_current = shell_info_all[path_params_current]

        # try:
            
        #     # special_path_sim
        #     shells = load_shells_for_variant(conf, path_sim=sim_current['path_sim'], filename_shells=filename_shells, variant=variant, check_existing=args.test)
        
        # except NoFileException as err:
            
        #     LOGGER.error(f'failed to load shells, err={err}')
        #     LOGGER.error(f'---> errors for index={index} variant={variant}, skipping...')
            
        #     continue



            # path_sim = utils_cosmogrid.get_sims_path(sim_params, id_sim=perms_info['id_sim'][i])
            # paths_shells.append(path_sim)

        # check if exists
        # filepath_out = get_filepath_projected_maps(dirpath_out, variant)
        # if args.resume:
        #     maps_file_ready = utils_maps.check_maps_completed(conf, filepath_out, nside)
        #     if maps_file_ready:
        #         LOGGER.critical(f'maps exist {filepath_out}')
        #         continue

    # load the shell groups and permutations
    # perms_info, shell_groups = utils_shells.load_permutation_index(filepath_perm_index)
    
    # get probe weights   

    # get output dir
    # dirpath_out = get_dirname_permuted_maps(args.dir_out, 
    #                                         cosmo_params=sim_params, 
    #                                         project_tag=conf['tag'], 
    #                                         id_perm=id_perm)

    # filename_kernels = get_filename_probe_kernels(dirpath_out)
    # shell_weights, probe_kernels = utils_maps.load_probe_weigths(filename_kernels, conf)
    
    # load_haloshells = check_if_load_haloshells(probe_kernels)
    # load_partshells = check_if_load_partshells(probe_kernels)


# def load_shells_for_variant(conf, path_sim, filename_shells, variant, check_existing):


#     shells = {}

#     if variant in ['v11dmo', 'v11dmb']:

#         filepath_shells_local = utils_maps.copy_cosmogrid_file(conf, path_sim, filename_shells, check_existing=check_existing, store_key='bary')
#         partshells =  utils_maps.load_v11_shells(filepath_shells_local, variant)
#         for k,s in partshells.items():
#             LOGGER.info(f'read shells {k} with size {s.shape} from {filepath_shells_local}  value lims=[{np.min(s):2.4e} {np.max(s):2.4e}]')
#         shells.update(partshells)


#     else:
        
#         raise Exception(f'unknown analysis variant {variant}')

#     return shells
    # # legacy for CosmoGridV1
    # if variant == 'baryonified512':
    #     filename_shells = 'baryonified_shells.npz'
    #     nside = 512
    
    # # legacy for CosmoGridV1
    # elif variant == 'nobaryons512':
    #     filename_shells = 'shells_nside=512.npz'
    #     nside = 512

    # # legacy for CosmoGridV1
    # elif variant == 'nobaryons2048':
    #     filename_shells = 'compressed_shells.npz'
    #     nside = 2048

# def check_if_load_haloshells(nz_info):

#     for nzi in nz_info:
#         if ('dh2' in nzi['probes']) or ('dh' in nzi['probes']): 
#             return True

#     return False


# def check_if_load_partshells(nz_info):

#     all_probes = []
#     for nzi in nz_info:
#         all_probes += nzi['probes']

#     all_probes = list(set(all_probes))
#     halo_probes = [h for h in all_probes if 'dh' in h]

#     if len(all_probes) == len(halo_probes): 
#             return False

#     return True





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


    # def get_shell_distances_old(path_logs):
    # """
    # Uses extracted files 'CosmoML.log' and 'baryonification_params.py'
    # :return shell_info_cov: shell info rec array with fields 'id', 'z_min', 'z_max', 'shell_cov', 'cov_inner', 'cov_outer'
    # """

    # from cosmogridv11 import baryonification
    # from cosmogridv11.baryonification import halocone


    # # get z bounds for shells
    # log_file = os.path.join(path_logs, "CosmoML.log")  # log file of the simulation
    # z_bounds = halocone.extract_redshift_bounds(log_file)
    # sorting = np.argsort(z_bounds[:,0])
    # z_bounds = z_bounds[sorting,:]
    # n_shells = len(z_bounds)
    # LOGGER.info(f"Extracted redshift bounds for n_shells={n_shells}")
    # LOGGER.debug("Extracted redshift bounds: " + str(z_bounds))

    # # load sim parameters and build cosmology
    # sys.path.insert(0,path_logs)
    # from baryonification_params import par
    # cosmo = baryonification.utils.build_cosmo(param=par)

    # # calcualte distances to shell boundaries

    # shell_info_cov = utils_arrays.zeros_rec(n_shells, columns=['shell_id:i4', 'lower_z', 'upper_z', 'shell_com', 'lower_com', 'upper_com'])
    # for i in range(len(z_bounds)):
    #     shell_info_cov[i] = i, z_bounds[i][0], z_bounds[i][1], *baryonification.halo_utils.get_shell_cov_dist(z_bounds[i], cosmo)


    # return shell_info_cov, par




    
    # # # divide the shells into groups that span a single box
    # # shell_groups = utils_shells.get_shell_groups(shell_info_current, n_max_replicas, Lbox=params_current['box_size_Mpc_over_h'])
    # # n_shell_groups = len(shell_groups)

    # # # calculate permutations list
    # # seed = conf['projection']['shell_perms_seed']+index_perm
    # # np.random.seed(seed)
    # # LOGGER.info(f'using random seed={seed} for shell permutations')
    # # list_perms_info = []

    # # output 
    # dirname_out = get_dirname_permuted_maps(dir_out=args.dir_out,
    #                                         project_tag=conf['tag'],
    #                                         cosmo_params=params_current,
    #                                         id_perm=index_perm)
    # utils_io.robust_makedirs(dirname_out)
    # filepath_perm_index = get_filepath_permutations_index(dirname_out)

    # # check if exists
    # perms_file_ready = False
    # if args.resume:
    
    #     LOGGER.info(f'resuming mode, file exists {filepath_perm_index}')
    #     perms_file_ready = utils_shells.check_perms_completed(filepath_perm_index)

    # # calculate and store the permutation sequence
    # if not perms_file_ready:
    
    #     perms_info = utils_shells.get_shell_permutation_sequence(n_shell_groups, n_sims_use)

    #     if args.test:
    #         LOGGER.warning('>>>>>>>>>>>>>>>> TEST mode, switching off permutations and shell shuffling!')
    #         perms_info[:] = 0, 0, False, False

    #     list_perms_info.append(perms_info)
    #     utils_shells.store_permutation_index(filepath_perm_index, perms_info, shell_groups)

    # # main magic - get the projected maps from permuted sims
    # files_maps, filename_kernels = project_single_permuted_sim(conf, args, filepath_perm_index, params_current, id_perm=index_perm, parslist_all=parslist_all)

    # # if needed, copy to external archive and remove
    # if args.dir_out_archive is not None:

    #     utils_io.archive_results(files_to_copy=[filepath_perm_index, filename_kernels] + files_maps,
    #                              dir_out=args.dir_out,
    #                              dir_out_archive=args.dir_out_archive)

# seed = conf['projection']['shell_perms_seed']+index_perm
    
    # n_sims_avail = np.count_nonzero(simslist_all['path_par']==path_params)
    # n_sims_use = min(float(n_max_sims_use), n_sims_avail)
    
    
# def project_permuted_sims(index, args, conf):
#     """
#     :param index: runs over cosmological parameters sets
#     """
    
#     raise Exception('not implemented yet')

# def project_single_permuted_sim(probe_kernels, shell_weights, perms_info, shell_groups, dirpath_out, conf, sim_params, seed_highz, parslist_all):
         
#     raise Exception('not implemented yet')
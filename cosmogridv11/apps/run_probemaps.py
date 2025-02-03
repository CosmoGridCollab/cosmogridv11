# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2022
author: Tomasz Kacprzak
"""

import os, warnings, argparse, h5py, numpy as np, time
from cosmogridv1 import utils_io, utils_logging, utils_config, utils_cosmogrid, utils_shells, utils_maps, utils_projection
from cosmogridv1.filenames import *
import healpy as hp
from cosmogridv1.copy_guardian import NoFileException

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

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
           'main_time_per_index':8, # hours
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
            res['pass'] = {'constraint': 'cpu', 'account': 'des', 'qos': 'shared'}
            res['main_nsimult'] = 200

        if os.environ['CLUSTER_NAME'] == 'euler':
            res['main_nsimult'] = 400
    
    return res


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
                files_out = project_permuted_sims(id_, args, conf)
            else:
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





def project_permuted_sims(index, args, conf):
    """
    :param index: runs over cosmological parameters sets
    """

    # SETUP

    # load parameter list
    permlist_all = utils_cosmogrid.load_permutations_list(conf)
    simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_baryonified_simulations_list(conf, set_type='all')
    LOGGER.info(f'loaded {len(parslist_all)} parameters and {len(simslist_all)} simulations')
    # simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_simulations_list(set_type='all')

    # index calculations
    perm_current = permlist_all[index]
    index_perm =  perm_current['id_perm']
    index_par = perm_current['row_id_parslist']
    params_current = parslist_all[index_par]
    shellinfo_current = shell_info_all[params_current['path_par']]
    n_sims_avail = np.count_nonzero(simslist_all['path_par']==params_current['path_par'])
    n_sims_use = min(float(conf['projection']['n_max_sims_use']), n_sims_avail)
    nz_info = conf['redshifts_nz']

    
    # get permutation indices range to run on
    dirname_out = get_dirname_permuted_maps(dir_out=args.dir_out,
                                            project_tag=conf['tag'],
                                            cosmo_params=params_current,
                                            id_perm=index_perm)
    utils_io.robust_makedirs(dirname_out)
    utils_io.ensure_permissions(dirname_out, verb=True)

    # start processing

    LOGGER.info(f"=============================> index={index} index_par={index_par} index_perm={index_perm} n_sims_use={n_sims_use} path_par={params_current['path_par']}")

    # KERNELS

    nz_info, w_shell = get_probe_kernels(nz_info, 
                                         params=params_current, 
                                         perm=perm_current, 
                                         shellinfo=shellinfo_current, 
                                         nside_out=int(conf['baryonification']['nside_out']), 
                                         redshift_error_method=conf['redshift_error_method'], 
                                         test=args.test)

    # SHELL GROUPS

    perms_info, shell_groups = get_shell_permutations(params=params_current, 
                                                      shellinfo=shellinfo_current,
                                                      n_sims_use=min(float(conf['projection']['n_max_sims_use']), n_sims_avail),
                                                      n_max_replicas=int(conf['projection']['n_max_replicas']), 
                                                      seed=conf['projection']['shell_perms_seed'] + index_perm,
                                                      test=args.test)

    # PROJECTED MAPS

    # files_maps, filename_kernels = project_single_permuted_sim(conf, args, filepath_perm_index, params_current, id_perm=index_perm, parslist_all=parslist_all)
    files_variants = project_single_permuted_sim(nz_info, w_shell, perms_info, shell_groups,
                                                 dirpath_out=dirname_out,
                                                 conf=conf, 
                                                 sim_params=params_current,
                                                 seed_highz=conf['projection']['highz_synfast_seed'] + params_current['id_param']*10000 + index_perm,
                                                 parslist_all=parslist_all)

    return files_variants


def project_single_permuted_sim(probe_kernels, shell_weights, perms_info, shell_groups, dirpath_out, conf, sim_params, seed_highz, parslist_all):
        
    # report
    LOGGER.info(f"=================>  path_par={sim_params['path_par']}")


    files_variants = []
    for variant in conf['analysis_variants']:

        LOGGER.info(f"============> creating permuted lightcone maps for variant={variant}, permutation id_sims={perms_info['id_sim']}")

        # get the right input and load
        filename_shells, nside = get_filename_shells_for_variant(variant)

        paths_shells = []
        for i, id_sim in enumerate(perms_info['id_sim']):
            path_sim = utils_cosmogrid.get_sims_path(sim_params, id_sim=perms_info['id_sim'][i])
            path_sim = os.path.join(path_sim, '', filename_shells)
            path_sim = path_sim.split('CosmoGrid/bary/')[1]
            path_sim = './' + path_sim
            paths_shells.append(path_sim)

        path_shells_local = utils_maps.copy_cosmogrid_file(conf, 
                                                           path_sim='', 
                                                           filename=paths_shells, 
                                                           check_existing=True, 
                                                           store_key='bary',
                                                           rsync_R=True)
        # loadd all the shells from different simulations according to the shell group permutation index
        shells_perm = {}
        success = True
        for i, id_sim in enumerate(perms_info['id_sim']):

            # load shells from the right file and select the shell needed
            LOGGER.info(f"======> loading sim {i+1}/{len(perms_info)}, id_sim={perms_info['id_sim'][i]}")
            path_sim = utils_cosmogrid.get_sims_path(sim_params, id_sim=perms_info['id_sim'][i])

            try:
                shells =  utils_maps.load_v11_shells(path_shells_local[i], variant)

            except NoFileException as err:
                LOGGER.error(f'failed to load shells, err={err}')
                LOGGER.error(f'---> errors for variant={variant}, skipping...')
                success = False
                break

            # select the relevant shell group according to perm indices
            select_group = shell_groups[i]['shell_id']
            LOGGER.info('select_group='+str(select_group))

            try:
                shells_group = select_shell_group(shells, select_group)
            except Exception as err:
                import pudb; pudb.set_trace();
                pass

            # random flips and rotations on the shell group
            for k,s in shells_group.items():
                shells_group[k] = utils_shells.add_flips_and_rots(s, perms_info[i])

            # store
            for k,s in shells_group.items():
                shells_perm.setdefault(k, [])
                shells_perm[k].append( shells_group[k] )
            del(shells)

        if not success:
            LOGGER.warning(f'unsuccessful shell group creation for variant {variant}, skipping..')
            continue

        # stack all shels to create the permuted lightcone
        shells = {}
        for k,s in shells_perm.items():
            shells[k] = np.concatenate(s, axis=0)
            LOGGER.info(f'stacked shell {k} groups size={shells[k].shape}')

        # main magic: project probes
        probe_maps = utils_projection.project_all_probes(shells, probe_kernels, shell_weights)

        # add high redshift shell using Gaussian Random Field from stored cls
        probe_maps = utils_projection.add_highest_redshift_shell(probe_maps, probe_kernels, sim_params, parslist_all, seed=seed_highz)

        # cell check
        probe_cells = check_cls_for_probes(probe_maps, probe_kernels, sim_params)    

        # output files and store
        filepath_out = get_filepath_projected_maps(dirpath_out, variant)
        LOGGER.info('storing maps')
        utils_maps.store_probe_maps(filepath_out, probe_maps, probe_cells=probe_cells, survey_mask=conf['projection']['survey_mask'], mode='w')
        LOGGER.info('storing kernels')
        utils_projection.store_probe_kernels(filepath_out, probe_kernels, shell_weights, mode='a')
        LOGGER.info('storing permutation indices')
        utils_shells.store_permutation_index(filepath_out, perms_info, shell_groups, mode='a')

        files_variants.append(filepath_out)

    # return files_variants, filename_kernels
    return files_variants


def project_single_sim(index, args, conf):


    simslist_all, parslist_all, shellinfo_all = utils_cosmogrid.get_baryonified_simulations_list(conf, set_type='all')
    sim_current = simslist_all[index]
    shellinfo_current = shellinfo_all[sim_current['path_par']]

    LOGGER.info(f"=============================> index={index} sim={sim_current['path_par']}")

    # get kernels
    nz_info = conf['redshifts_nz']
    nz_info, w_shell = get_probe_kernels(nz_info, 
                                         params=sim_current, 
                                         perm=None, 
                                         shellinfo=shellinfo_current, 
                                         nside_out=int(conf['baryonification']['nside_out']), 
                                         test=args.test)

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
        probe_maps = utils_projection.project_all_probes(shells, nz_info, w_shell)

        # s_ = check_cls_for_probes(probe_maps, nz_info, sim_current, plot=True)

        # add high redshift shell using Gaussian Random Field from stored cls
        probe_maps = utils_projection.add_highest_redshift_shell(probe_maps, nz_info, sim_params=sim_current, parslist_all=parslist_all, seed=index+1231)
        del(shells)

        probe_cells = check_cls_for_probes(probe_maps, nz_info, sim_current)

        # output files and store
        filepath_out = get_filepath_projected_maps(dirpath_out, variant)
        LOGGER.info('storing maps')
        utils_maps.store_probe_maps(filepath_out, probe_maps, probe_cells=probe_cells, survey_mask=conf['projection']['survey_mask'], mode='w')
        LOGGER.info('storing kernels')
        utils_projection.store_probe_kernels(filepath_out, nz_info, w_shell, mode='a')

        files_variants.append(filepath_out)

    return files_variants



def get_auto_cell(probe, z, nz, ell, cosmo):

    import pyccl as ccl
    import pyccl.nl_pt as pt

    if probe == 'kg':
        
        tracer = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))
        p_of_k_a = 'delta_matter:delta_matter'

    elif probe == 'ia':

        ptc = pt.EulerianPTCalculator(cosmo=cosmo, with_IA=True)
        c1, c1delta, c2 = pt.tracers.translate_IA_norm(cosmo=cosmo, z=z, a1=1, a1delta=0, a2=0)
        p_of_k_a = ptc.get_biased_pk2d(pt.PTIntrinsicAlignmentTracer(c1=(z, c1), c2=(z, c2), cdelta=(z, c1delta)))
        tracer = ccl.WeakLensingTracer(cosmo, dndz=(z, nz), has_shear=False, ia_bias=(z, np.ones_like(z)), use_A_ia=False)

    elif probe == 'dg':

        ptc = pt.EulerianPTCalculator(cosmo=cosmo, with_NC=True)
        p_of_k_a = ptc.get_biased_pk2d(pt.PTNumberCountsTracer(b1=1, b2=0))
        tracer = ccl.NumberCountsTracer(cosmo, dndz=(z, nz), bias=(z, np.ones_like(z)), has_rsd=False, mag_bias=None)

    elif probe == 'kcmb':

        tracer = ccl.CMBLensingTracer(cosmo, z_source=1100, n_samples=100)
        p_of_k_a = 'delta_matter:delta_matter'

    return ccl.angular_cl(cosmo, tracer, tracer, ell,  p_of_k_a=p_of_k_a)


def check_cls_for_probes(probe_maps, nz_info, params, plot=False):

    import pyccl as ccl

    cosmo = ccl.Cosmology(Omega_c=params['O_cdm'], 
                          Omega_b=params['Ob'], 
                          h=params['H0']/100, 
                          A_s=params['As'], 
                          n_s=params['ns'],
                          w0= params['w0'],
                          Neff=3.15, 
                          m_nu=0.06)

    probe_cells = {}
    for sample in nz_info:

        n = sample['name']
        
        for probe in sample['probes']:

            probe_cells.setdefault(probe, {})
            
            z, nz = sample['nz'].T
            m = probe_maps[probe][n]
            lmax = 3*hp.npix2nside(len(m))-1
            ell = np.arange(0, lmax+1)

            cell_ccl = get_auto_cell(probe, z, nz, ell, cosmo)
            cell_map = hp.alm2cl(hp.map2alm(m))
            m_ccl = hp.synfast(cell_ccl, hp.npix2nside(len(m)), pixwin=True)
            cell_map_ccl = hp.alm2cl(hp.map2alm(m_ccl))

            err = np.median(cell_map_ccl/cell_map-1)
            LOGGER.info(f"cl for {n} {probe} map-vs-ccl median error {err:2.2e}")
            if np.abs(err)>0.1:
                LOGGER.warning('================> high error in Cell between map and ccl !!!')
            
            if plot:    
                import matplotlib; matplotlib.use('module://matplotlib-backend-sixel'); import matplotlib.pyplot as plt
                s_ = np.s_[100:1000]
                plt.figure()
                plt.plot(ell[s_], cell_map_ccl[s_] * ell[s_] * (ell[s_]+1) / 2 / np.pi, label=f'ccl {probe} {n}')
                plt.plot(ell[s_], cell_map[s_]     * ell[s_] * (ell[s_]+1) / 2 / np.pi, label=f'map {probe} {n}')
                plt.legend()
                plt.show()

            probe_cells[probe][n] =  (ell, cell_ccl, cell_map_ccl, cell_map)

    return probe_cells



def get_probe_kernels(nz_info, params, shellinfo, nside_out, perm=None, redshift_error_method='fishbacher', test=False):

    from cosmogridv1 import utils_redshift, utils_projection

    n_samples = len(nz_info)
    LOGGER.info(f'using {n_samples} redshift bins')

    # test mode?
    if test:
        LOGGER.warning('------------> TEST MODE')
        n_samples = 1

    astropy_cosmo = utils_cosmogrid.build_cosmo(params)
    LOGGER.info(astropy_cosmo)

    # get simulation parameters
    sim_params = dict(n_pix=hp.nside2npix(nside_out), 
                      n_particles=params['n_particles'], 
                      box_size=params['box_size_Mpc_over_h'])

    if test:
        LOGGER.warning('------------> TEST MODE')
        shellinfo = shellinfo[:4]

    # get shell weights
    w_shell = utils_projection.get_kernel_shell(shellinfo, astropy_cosmo, sim_params, kw_ufalcon, test=test)

    # iterate over redshift bins
    for j, sample in enumerate(nz_info):

        LOGGER.info(f'=======> sample {j}/{len(nz_info)} {sample["name"]} probes {sample["probes"]}')

        # for the case when we do not do shell permutations, ignore redshift perturbations

        if 'file' in sample:
        
            if perm is None:

                sample['nz'] = utils_maps.load_nz(sample['file'], z_max=sample['z_max'])
                LOGGER.warning('----> running shell projection without permutation, ignoring redshfit perturbations')

            else:

                redshift_mod = [perm[sample['name']+'__delta_meanz'], perm[sample['name']+'__delta_sigmaz']]
                LOGGER.info(f'redshift delta mean {redshift_mod[0]: 2.4e} delta sigma {redshift_mod[1]: 2.4e}')

                sample['nz'] = utils_redshift.modify_nz(nz=utils_maps.load_nz(sample['file'], z_max=sample['z_max']), 
                                                            redshift_params=redshift_mod, 
                                                            tag=sample['name'], 
                                                            method=redshift_error_method)

        # get weights for probes
        for probe in sample['probes']:

            LOGGER.info(f'getting projection kernels for probe {probe}')

            # main magic: project probes
            utils_projection.probe_kernel_funcs[probe](sample, shellinfo, astropy_cosmo, sim_params, kw_ufalcon, test=test)
    
    return nz_info, w_shell


def get_shell_permutations(params, shellinfo, n_sims_use, n_max_replicas, seed, test=False):

    # divide the shells into groups that span a single box
    shell_groups = utils_shells.get_shell_groups(shellinfo, n_max_replicas, Lbox=params['box_size_Mpc_over_h'])
    n_shell_groups = len(shell_groups)

    # calculate permutations list
    np.random.seed(seed)
    LOGGER.info(f'using random seed={seed} for shell permutations')

    perms_info = utils_shells.get_shell_permutation_sequence(n_shell_groups, n_sims_use)

    if test:
        LOGGER.warning('>>>>>>>>>>>>>>>> TEST mode, switching off permutations and shell shuffling!')
        perms_info[:] = 0, 0, False, False

    return perms_info, shell_groups





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

    # from cosmogridv1 import baryonification
    # from cosmogridv1.baryonification import halocone


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
    
    
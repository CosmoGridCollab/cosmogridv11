
# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2022
author: Tomasz Kacprzak
"""

import os, sys, warnings, argparse, h5py, numpy as np, time, logging, itertools, shutil
from cosmogridv1 import utils_io, utils_logging, utils_config, utils_cosmogrid, utils_maps
from cosmogridv1.filenames import *
from UFalcon import probe_weights
import healpy as hp

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
                        help='configuration yaml file')
    parser.add_argument('--dir_out', type=str, required=True, 
                        help='output dir for the results')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    args, _ = parser.parse_known_args(args)

    utils_logging.set_all_loggers_level(args.verbosity)

    # get absolute paths
    args.config = utils_io.get_abs_path(args.config)
    args.dir_out = utils_io.get_abs_path(args.dir_out)


    return args


def resources(args):
    
    res = {'main_memory':8000,
           'main_time_per_index':5./60., # hours
           'main_scratch':6500,
           'merge_memory':64000,
           'merge_time':24}
           # 'pass':{'constraint': 'knl', 'account': 'des', 'qos': 'regular'}} # Cori
    
    return res


def modify_nz(nz, redshift_params, tag):

    from cosmogridv1.utils_redshift import shift_and_stretch

    if np.all(redshift_params==0):
        
        return nz 

    else:


        z_ = nz[:,0]
        nz_ = nz[:,1]/np.sum(nz[:,1])

        sigma_z =  np.sqrt(np.sum( nz_ * (z_-np.sum(nz_*z_))**2 ))
        mean_z = np.sum(nz_ * z_)

        shift = redshift_params[0]
        stretch = (sigma_z-redshift_params[1])/sigma_z
        nz_mod = shift_and_stretch(nz=[nz], 
                                   z_bias=[shift], 
                                   z_sigma=[stretch], # multiplicative stretch used in utils_redshift, config interface uses additive stretch
                                   normalize=True)[0]

        LOGGER.info(f'bin={tag} mean_z={mean_z} sigma_z={sigma_z} shift={shift} stretch={stretch}')

    return nz_mod


def main(indices, args):
    """
    Project shells using probe weights computed realier.
    Code lifted from Janis Fluri's Kids1000 analysis.
    https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/map_projection/patch_generation/project_patches.py#L1
    """

    from cosmogridv1.copy_guardian import NoFileException

    args = setup(args)
    
    # loop over sims 

    for index in indices: 
        
        # re-load config
        conf = utils_config.load_config(args.config)

        if conf['projection']['shell_perms'] == False:

            kernels_single_sim(index, args, conf)

        else:
            
            kernels_permuted_sim(index, args, conf)


        yield index

def missing(indices, args):

    # get basic config
    args = setup(args)
    conf = utils_config.load_config(args.config)

    # load redshift perturbations
    permlist_all = load_redshift_perturb_list(conf)

    # get simulation list
    simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_simulations_list(set_type='all')

    list_missing = []
    for index in indices: 

        if conf['projection']['shell_perms'] == False:

            raise Exception('not implemented yet')

        else:

            index_perm =  permlist_all['id_perm'][index]
            index_par = permlist_all['row_id_parslist'][index]
            cosmo_params = parslist_all[index_par]

            dirpath_out = get_dirname_permuted_maps(args.dir_out, 
                                                    cosmo_params=cosmo_params, 
                                                    project_tag=conf['tag'], 
                                                    id_perm=index_perm)
            filename_out = get_filename_probe_weights(dirpath_out)

            if os.path.isfile(filename_out):

                LOGGER.debug(f'OK     index {index: 6d} file {filename_out}')

            else:

                LOGGER.info(f'missing index {index: 6d} file {filename_out}')
                list_missing.append(index)

    LOGGER.info(f'missing {len(list_missing)} indides')
    LOGGER.info(list_missing)

    return list_missing



def load_redshift_perturb_list(conf):

    assert 'redshift_perturbations_list' in conf['paths'], 'path to shell permutation list missing, add config[paths][redshift_perturbations_list]'
    filename_permlist = str(conf['paths']['redshift_perturbations_list'])
    permlist = np.load(filename_permlist)
    LOGGER.info(f'loaded {filename_permlist} with {len(permlist)} permutations')
    
    return permlist


def kernels_single_sim(index, args, conf):

    from UFalcon import probe_weights

    # make tf records

    args = setup(args)
    conf = utils_config.load_config(args.config)
    
    # get simulation list
    simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_simulations_list(set_type='all')

    # make output dirs
    utils_io.robust_makedirs(args.dir_out)

    # get redshift bins
    nz_info = conf['redshifts_nz']
    n_redshift_bins = len(nz_info)
    LOGGER.info(f'using {n_redshift_bins} redshift bins')
    for i, p in enumerate(nz_info):
        LOGGER.info(f'bin {i+1}/{n_redshift_bins} {p}')

    if args.test:
        LOGGER.warning('------------> TEST MODE')
        n_redshift_bins = 1


    # get cosmology
    cosmo_params = parslist_all[index]
    shell_info = shell_info_all[cosmo_params['path_par']]

    LOGGER.info(f"===================> index={index} path={cosmo_params['path_par']}")

    astropy_cosmo = utils_cosmogrid.build_cosmo(cosmo_params)
    LOGGER.info(astropy_cosmo)

    if args.test:
        LOGGER.warning('------------> TEST MODE')
        shell_info = shell_info[:4]

    # output container
    list_w = {}
        
    w_shell = get_kernel_shell(shell_info, astropy_cosmo=astropy_cosmo, kw_ufalcon=kw_ufalcon, test=args.test)

    # iterate over redshift bins
    for j in range(n_redshift_bins):

        # find nz 
        if 'file' in nz_info[j]:
        
            nz_info[j]['nz'] = utils_maps.load_nz(nz_info[j]['file'], z_max=nz_info[j]['z_max'])


        for probe in nz_info[j]['probes']:

            if probe == 'kd':
                
                if 'kg' not in nz_info[j]['probes']:
                    raise Exception('lensing probe (kg) is required for computation of the source clustering (kd) component, add kg to the probes list')

                LOGGER.info('no need for source clustering kernel calculation')

                continue
            
            nz_info = probe_kernel_funcs[probe](j, n_redshift_bins, nz_info, shell_info=shell_info, astropy_cosmo=astropy_cosmo, kw_ufalcon=kw_ufalcon, test=args.test)

            store_probe_weights(dir_out=args.dir_out, 
                                cosmo_params=cosmo_params, 
                                nz_info=nz_info,
                                w_shell=w_shell)
    




def kernels_permuted_sim(index, args, conf):

    from UFalcon import probe_weights

    # get simulation list
    simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_simulations_list(set_type='all')

    # load redshift perturbations
    permlist_all = load_redshift_perturb_list(conf)

    # get redshift bins
    nz_info = conf['redshifts_nz']
    n_redshift_bins = len(nz_info)
    LOGGER.info(f'using {n_redshift_bins} redshift bins')
    for i, p in enumerate(nz_info):
        LOGGER.info(f'bin {i+1}/{n_redshift_bins} {p}')

    if args.test:
        LOGGER.warning('------------> TEST MODE')
        n_redshift_bins = 1

    index_perm =  permlist_all['id_perm'][index]
    index_par = permlist_all['row_id_parslist'][index]

    # get cosmology
    cosmo_params = parslist_all[index_par]
    shell_info = shell_info_all[cosmo_params['path_par']]
    astropy_cosmo = utils_cosmogrid.build_cosmo(cosmo_params)
    LOGGER.info(astropy_cosmo)

    # make permutation dir
    dirpath_out = get_dirname_permuted_maps(args.dir_out, 
                                            cosmo_params=cosmo_params, 
                                            project_tag=conf['tag'], 
                                            id_perm=index_perm)

    utils_io.robust_makedirs(dirpath_out)

    LOGGER.info(f"===================> index={index} index_par={index_par} index_perm={index_perm} path={cosmo_params['path_par']}")


    if args.test:
        LOGGER.warning('------------> TEST MODE')
        shell_info = shell_info[:4]

    # output container
    list_w = {}
        
    w_shell = get_kernel_shell(shell_info, astropy_cosmo=astropy_cosmo, kw_ufalcon=kw_ufalcon, test=args.test)

    # iterate over redshift bins
    for j in range(n_redshift_bins):

        LOGGER.info(f'=======> redshift bin {j} {nz_info[j]["name"]} probes {nz_info[j]["probes"]}')

        # find nz 
        if 'file' in nz_info[j]:

            redshift_mod = [permlist_all[index][nz_info[j]['name']+'__delta_meanz'], permlist_all[index][nz_info[j]['name']+'__delta_sigmaz']]
            LOGGER.info(f'redshift delta mean {redshift_mod[0]} delta sigma {redshift_mod[1]}')

            nz_info[j]['nz'] = modify_nz(nz=utils_maps.load_nz(nz_info[j]['file'], z_max=nz_info[j]['z_max']), 
                                         redshift_params=redshift_mod, 
                                         tag=nz_info[j]['name'])

        for probe in nz_info[j]['probes']:

            if probe == 'kd':
                
                if 'kg' not in nz_info[j]['probes']:
                    raise Exception('lensing probe (kg) is required for computation of the source clustering (kd) component, add kg to the probes list')

                LOGGER.info('no need for source clustering kernel calculation')

                continue

            nz_info = probe_kernel_funcs[probe](j, n_redshift_bins, nz_info, shell_info=shell_info, astropy_cosmo=astropy_cosmo, kw_ufalcon=kw_ufalcon, test=args.test)

    store_probe_weights(dir_out=dirpath_out, 
                        cosmo_params=cosmo_params, 
                        nz_info=nz_info,
                        w_shell=w_shell)


    # additionally store probe weights for the 0-th permutation (no redshift perturb) in the main parameter dir
    if index_perm == 0:

        dirpath_base = get_dirname_probe_weights(args.dir_out, 
                                                 cosmo_params=cosmo_params, 
                                                 project_tag=conf['tag'])
        
        store_probe_weights(dir_out=dirpath_base, 
                            cosmo_params=cosmo_params, 
                            nz_info=nz_info,
                            w_shell=w_shell)

            # inerate over probes
            # for probe in nz_info[j]['probes']:

            #     if probe == 'kg':

            #     #     nz_info[j][f'w_kg'] = np.zeros(len(shell_info))

            #     #     weight_generator_kg = probe_weights.Continuous_lensing(n_of_z=nz_info[j]['nz'], 
            #     #                                                            shift_nz=shift_nz, 
            #     #                                                            fast_mode=2,
            #     #                                                            **kw_ufalcon)
                    
            #     #     prefactor_kappa = probe_weights.kappa_prefactor_nosim(cosmo=astropy_cosmo)

            #     #     for s in LOGGER.progressbar(range(len(shell_info)), desc=f'bin={j}/{n_redshift_bins} probe={probe}'):

            #     #         if check_zbound(shell_info, s, z_lim_up, probe):

            #     #             w_kg = weight_generator_kg(z_low=shell_info['lower_z'][s], 
            #     #                                        z_up=shell_info['upper_z'][s], 
            #     #                                        cosmo=astropy_cosmo)

            #     #             nz_info[j]['w_kg'][s] = w_kg * prefactor_kappa


            #         nz_info[j][f'w_kg_split'] = np.zeros((len(shell_info), len(shell_info)))
                    
            #         weight_generator_kg = probe_weights.Continuous_lensing(n_of_z=nz_info[j]['nz'], 
            #                                                                shift_nz=shift_nz, 
            #                                                                fast_mode=2,
            #                                                                **kw_ufalcon)
                                        
            #         prefactor_kappa = probe_weights.kappa_prefactor_nosim(cosmo=astropy_cosmo)

            #         for s1 in LOGGER.progressbar(range(len(shell_info)), desc=f'bin={j}/{n_redshift_bins} probe={probe}', at_level='info'):

            #             if check_zbound(shell_info, s1, z_lim_up, probe):

            #                 for s2 in range(s1, len(shell_info)):

            #                     w_kg = weight_generator_kg(z_low=shell_info['lower_z'][s1], 
            #                                                z_up=shell_info['upper_z'][s1],
            #                                                source_z_lims=[shell_info['lower_z'][s2], shell_info['upper_z'][s2]],
            #                                                cosmo=astropy_cosmo)

            #                     nz_info[j]['w_kg_split'][s1, s2] = w_kg * prefactor_kappa

            #             if args.test:
    
            #                 w_kg_full = weight_generator_kg(z_low=shell_info['lower_z'][s1], 
            #                                                 z_up=shell_info['upper_z'][s1],
            #                                                 cosmo=astropy_cosmo) * prefactor_kappa
            #                 w_kg_split = np.sum(nz_info[j]['w_kg_split'], axis=1)
            #                 LOGGER.debug('source clustering: checking split lensing kernel integration, {} split={:2.5e} full={:2.5e} diff={:2.5e}'.format(s1, w_kg_split[s1], w_kg_full, w_kg_split[s1]-w_kg_full))

            #         nz_info[j]['w_kg'] = np.sum(nz_info[j][f'w_kg_split'], axis=1)
            #         nz_info[j]['nz_norm'] = weight_generator_kg.nz_norm

            #     if probe == 'dg':

            #         nz_info[j][f'w_dg'] = np.zeros(len(shell_info))

            #         weight_generator_dg = probe_weights.Continuous_clustering(n_of_z=nz_info[j]['nz'],
            #                                                                   shift_nz=shift_nz,
            #                                                                   **kw_ufalcon)

            #         prefactor_delta = probe_weights.delta_prefactor_nosim(cosmo=astropy_cosmo)

            #         for s in LOGGER.progressbar(range(len(shell_info)), desc=f'bin={j}/{n_redshift_bins} probe={probe}'):

            #             if check_zbound(shell_info, s, z_lim_up, probe):

            #                 w_dg = weight_generator_dg(z_low=shell_info['lower_z'][s], 
            #                                            z_up=shell_info['upper_z'][s], 
            #                                            cosmo=astropy_cosmo,
            #                                            lin_bias=1)

            #                 nz_info[j]['w_dg'][s] = w_dg * prefactor_delta
                    
            #         nz_info[j]['nz_norm'] = weight_generator_dg.nz_norm

            #     elif probe == 'ia':

            #         nz_info[j][f'w_ia'] = np.zeros(len(shell_info))

            #         weight_generator_ia = probe_weights.Continuous_intrinsic_alignment(n_of_z=nz_info[j]['nz'], 
            #                                                                            shift_nz=shift_nz, 
            #                                                                            IA=1.0, 
            #                                                                            **kw_ufalcon)
                    
            #         prefactor_delta = probe_weights.delta_prefactor_nosim(cosmo=astropy_cosmo)

            #         for s in LOGGER.progressbar(range(len(shell_info)), desc=f'bin={j}/{n_redshift_bins} probe={probe}'):

            #             if check_zbound(shell_info, s, z_lim_up, probe):

            #                 w_ia = weight_generator_ia(z_low=shell_info['lower_z'][s], 
            #                                            z_up=shell_info['upper_z'][s], 
            #                                            cosmo=astropy_cosmo)

            #                 nz_info[j]['w_ia'][s] = w_ia * prefactor_delta
                    
            #         nz_info[j]['nz_norm'] = weight_generator_ia.nz_norm

            #     elif probe == 'kd':

            #         prefactor_delta = probe_weights.delta_prefactor_nosim(cosmo=astropy_cosmo)

            #         nz_info[j][f'w_kd'] = np.zeros(len(shell_info))

            #         weight_generator_kd = probe_weights.Dirac_clustering()

            #         for s in LOGGER.progressbar(range(len(shell_info)), desc=f'bin={j}/{n_redshift_bins} probe={probe}'):

            #             if check_zbound(shell_info, s, z_lim_up, probe):

            #                 w_kd = weight_generator_kd(z_low=shell_info['lower_z'][s], 
            #                                            z_up=shell_info['upper_z'][s], 
            #                                            cosmo=astropy_cosmo,
            #                                            lin_bias=1)

            #                 nz_info[j][f'w_kd'][s] = w_kd * prefactor_delta


            #     elif probe == 'k_cmb':

            #         nz_info[j][f'w_k_cmb'] = np.zeros(len(shell_info))

            #         weight_generator_kg = probe_weights.Dirac_lensing(z_source=nz_info[j]['z_cmb'])

            #         prefactor_kappa = probe_weights.kappa_prefactor_nosim(cosmo=astropy_cosmo)

            #         for s in LOGGER.progressbar(range(len(shell_info)), desc=f'bin={j}/{n_redshift_bins} probe={probe}'):

            #             w_k_cmb = weight_generator_kg(z_low=shell_info['lower_z'][s], 
            #                                           z_up=shell_info['upper_z'][s], 
            #                                           cosmo=astropy_cosmo)

            #             nz_info[j]['w_k_cmb'][s] = w_k_cmb * prefactor_kappa


            #     elif probe == 'd_isw':

            #         raise Exception('ISW is not implemented yet')


                # else:
                    
                #     nz_info[j][f'w_{probe}'] = np.zeros(len(shell_info))

                #     for s in  LOGGER.progressbar(range(len(shell_info)), at_level='info', desc=f'creating weights for z_bin={j+1}/{n_redshift_bins}'):

                #         LOGGER.debug(f'========== shell {s: 4d}/{len(shell_info)}')

                #         if z_lim_up < shell_info['lower_z'][s]:

                #             LOGGER.debug(f"shell outside redshift range max_z_bin={z_lim_up:2.4f} min_z_shell={shell_info['lower_z'][s]:2.4f}")

                #         else:

                #             LOGGER.debug(f'init weights')

                                

                            # weight_generator_ia = probe_weights.Continuous_intrinsic_alignment(n_of_z=nz_info[j]['nz'], 
                        #                                                                    shift_nz=shift_nz, 
                        #                                                                    IA=1.0, 
                        #                                                                    **kw_ufalcon)

                        # weight_generator_dg = probe_weights.Continuous_clustering(n_of_z=nz_info[j]['nz'],
                        #                                                           shift_nz=shift_nz,
                        #                                                           **kw_ufalcon)


                        # LOGGER.debug(f'call kappa prefactor')

                        # # prefactor_kappa_old = lensing_weights.kappa_prefactor(n_pix=hp.nside2npix(n_side), 
                        #                                                       n_particles=n_particles**3,  
                        #                                                       boxsize=box_size / (astropy_cosmo.H0.value / 100) / 1000, 
                        #                                                       cosmo=astropy_cosmo)

                        # n_particles = 832
                        # n_side = 512
                        # box_size = 900

                        # prefactor_kappa_old = probe_weights.kappa_prefactor(n_pix=hp.nside2npix(n_side),
                        #                                                     n_particles=n_particles**3, 
                        #                                                     boxsize=box_size / (astropy_cosmo.H0.value / 100) / 1000, 
                        #                                                     cosmo=astropy_cosmo)
                        
                        
                        # LOGGER.debug(f'call delta prefactor')


                        # prefactor_delta_old = probe_weights.delta_prefactor(n_pix=hp.nside2npix(n_side),
                        #                                                     n_particles=n_particles**3, 
                        #                                                     boxsize=box_size / (astropy_cosmo.H0.value / 100) / 1000, 
                        #                                                     cosmo=astropy_cosmo)
                        
                        # prefactor_delta = probe_weights.delta_prefactor_nosim(cosmo=astropy_cosmo)

                        # prefactor_sim = probe_weights.sim_spec_prefactor(n_pix=hp.nside2npix(n_side), 
                        #                                                  box_size=box_size, # input in Mpc
                        #                                                  n_particles=n_particles**3)

                        # LOGGER.debug(f'call kg weights')

                        # w_kg_old = weight_generator_kg_old(z_low=shell_info['lower_z'][s], 
                        #                                    z_up=shell_info['upper_z'][s], 
                        #                                    cosmo=astropy_cosmo)

                        # w_kg = weight_generator_kg(z_low=shell_info['lower_z'][s], 
                        #                            z_up=shell_info['upper_z'][s], 
                        #                            cosmo=astropy_cosmo)
                        
                        # LOGGER.debug(f'call ia weights')

                        # w_ia = weight_generator_ia(z_low=shell_info['lower_z'][s], 
                        #                            z_up=shell_info['upper_z'][s], 
                        #                            cosmo=astropy_cosmo)
                        
                        # LOGGER.debug(f'call dg weights')

                        # w_dg = weight_generator_dg(z_low=shell_info['lower_z'][s], 
                        #                            z_up=shell_info['upper_z'][s], 
                        #                            cosmo=astropy_cosmo,
                        #                            lin_bias=1)

                    # nz_info[j]['w_kg_old'][s] = w_kg_old * prefactor_kappa_old
                    # nz_info[j]['w_kg'][s] = w_kg * prefactor_kappa
                    # nz_info[j]['w_ia'][s] = w_ia * prefactor_delta
                    # nz_info[j]['w_dg'][s] = w_dg * prefactor_delta

        #     store_probe_weights(dir_out=args.dir_out, 
        #                         project_tag=conf['tag'], 
        #                         cosmo_params=cosmo_params, 
        #                         nz_info=nz_info)
        
        # yield index

    
def store_probe_weights(dir_out, cosmo_params, nz_info, w_shell):

    all_probes = ['kg_split', 'kg', 'ia', 'dg', 'k_cmb']

    filename_out = get_filename_probe_weights(dir_out)
    dataset_name = lambda nzi, probe: f"{probe}/{nzi['name']}"

    datasets_stored = []
    with h5py.File(filename_out, 'w') as f:

        f.create_dataset(name='w_shell', data=w_shell, compression='lzf', shuffle=True)

        for nzi in nz_info:
            for probe in all_probes:
                try:
                    d = dataset_name(nzi, probe)
                    f.create_dataset(name=d, data=nzi[f'w_{probe}'], compression='lzf', shuffle=True)
                    if 'nz_norm' in nzi.keys():
                        f[d].attrs['nz_norm'] = nzi['nz_norm']

                    datasets_stored.append(d)
                except Exception as err:
                    LOGGER.debug(f'failed to store dataset {d} err={str(err)}')


    LOGGER.info(f'stored {filename_out} with datasets {str(datasets_stored)}')


def check_zbound(shell_info, s, z_lim_up, probe):

    if z_lim_up < shell_info['lower_z'][s]:
                       
        LOGGER.debug(f"shell outside redshift range max_z_bin={z_lim_up:2.4f} min_z_shell={shell_info['lower_z'][s]:2.4f}")
        return False

    else:

        LOGGER.debug(f'========== probe={probe} shell {s: 4d}/{len(shell_info)}')
        return True



#################################################################################################
#################################################################################################
###
### Kernel functions
###
#################################################################################################
#################################################################################################


def get_kernel_shell(shell_info, astropy_cosmo, kw_ufalcon, test=False):
    
    z_lim_up = kw_ufalcon['z_lim_up']
                    
    prefactor_delta = probe_weights.delta_prefactor_nosim(cosmo=astropy_cosmo)

    w_shell = np.zeros(len(shell_info))

    weight_generator_shell = probe_weights.Dirac_clustering()

    for s in range(len(shell_info)):

        if check_zbound(shell_info, s, z_lim_up, probe='shell weight'):

            w = weight_generator_shell(z_low=shell_info['lower_z'][s], 
                                             z_up=shell_info['upper_z'][s], 
                                             cosmo=astropy_cosmo,
                                             lin_bias=1)

            w_shell[s] = w * prefactor_delta

    return w_shell



def get_kernel_kg(j, n_redshift_bins, nz_info, shell_info, astropy_cosmo, kw_ufalcon, test=False):

    probe = 'kg'
    z_lim_up = kw_ufalcon['z_lim_up']

    nz_info[j][f'w_kg_split'] = np.zeros((len(shell_info), len(shell_info)))
    
    weight_generator_kg = probe_weights.Continuous_lensing(n_of_z=nz_info[j]['nz'], 
                                                           fast_mode=2,
                                                           **kw_ufalcon)
    weight_generator_kg.set_cosmo(cosmo=astropy_cosmo)
                        
    prefactor_kappa = probe_weights.kappa_prefactor_nosim(cosmo=astropy_cosmo)

    # lens shells
    for s1 in LOGGER.progressbar(range(len(shell_info)), desc=f'bin={j+1}/{n_redshift_bins} probe={probe}', at_level='info'):

        if check_zbound(shell_info, s1, z_lim_up, probe):

            # source shells
            for s2 in range(s1, len(shell_info)):

                w_kg = weight_generator_kg(z_low=shell_info['lower_z'][s1], 
                                           z_up=shell_info['upper_z'][s1],
                                           source_z_lims=[shell_info['lower_z'][s2], shell_info['upper_z'][s2]])

                nz_info[j]['w_kg_split'][s1, s2] = w_kg * prefactor_kappa

        if test:

            w_kg_full = weight_generator_kg(z_low=shell_info['lower_z'][s1], 
                                            z_up=shell_info['upper_z'][s1]) * prefactor_kappa
            w_kg_split = np.sum(nz_info[j]['w_kg_split'], axis=1)
            LOGGER.debug('source clustering: checking split lensing kernel integration, {} split={:2.5e} full={:2.5e} diff={:2.5e}'.format(s1, w_kg_split[s1], w_kg_full, w_kg_split[s1]-w_kg_full))

    nz_info[j]['w_kg'] = np.sum(nz_info[j][f'w_kg_split'], axis=1)
    nz_info[j]['nz_norm'] = weight_generator_kg.nz_norm

    return nz_info


def get_kernel_dg(j, n_redshift_bins, nz_info, shell_info, astropy_cosmo, kw_ufalcon, test=False):

    probe = 'dg'
    z_lim_up = kw_ufalcon['z_lim_up']

    nz_info[j][f'w_dg'] = np.zeros(len(shell_info))

    weight_generator_dg = probe_weights.Continuous_clustering(n_of_z=nz_info[j]['nz'],
                                                              **kw_ufalcon)

    prefactor_delta = probe_weights.delta_prefactor_nosim(cosmo=astropy_cosmo)

    for s in LOGGER.progressbar(range(len(shell_info)), desc=f'bin={j+1}/{n_redshift_bins} probe={probe}'):

        if check_zbound(shell_info, s, z_lim_up, probe):

            w_dg = weight_generator_dg(z_low=shell_info['lower_z'][s], 
                                       z_up=shell_info['upper_z'][s], 
                                       cosmo=astropy_cosmo,
                                       lin_bias=1)

            nz_info[j]['w_dg'][s] = w_dg * prefactor_delta
    
    nz_info[j]['nz_norm'] = weight_generator_dg.nz_norm

    return nz_info

def get_kernel_ia(j, n_redshift_bins, nz_info, shell_info, astropy_cosmo, kw_ufalcon, test=False):
    
    probe = 'ia'
    z_lim_up = kw_ufalcon['z_lim_up']

    nz_info[j][f'w_ia'] = np.zeros(len(shell_info))

    weight_generator_ia = probe_weights.Continuous_intrinsic_alignment(n_of_z=nz_info[j]['nz'], 
                                                                       IA=1.0, 
                                                                       **kw_ufalcon)
    
    prefactor_delta = probe_weights.delta_prefactor_nosim(cosmo=astropy_cosmo)

    for s in LOGGER.progressbar(range(len(shell_info)), desc=f'bin={j+1}/{n_redshift_bins} probe={probe}'):

        if check_zbound(shell_info, s, z_lim_up, probe):

            w_ia = weight_generator_ia(z_low=shell_info['lower_z'][s], 
                                       z_up=shell_info['upper_z'][s], 
                                       cosmo=astropy_cosmo)

            nz_info[j]['w_ia'][s] = w_ia * prefactor_delta
    
    nz_info[j]['nz_norm'] = weight_generator_ia.nz_norm

    return nz_info

def get_kernel_k_cmb(j, n_redshift_bins, nz_info, shell_info, astropy_cosmo, kw_ufalcon, test=False):

    probe = 'k_cmb'
    z_lim_up = kw_ufalcon['z_lim_up']

    nz_info[j][f'w_k_cmb'] = np.zeros(len(shell_info))

    weight_generator_kg = probe_weights.Dirac_lensing(z_source=nz_info[j]['z_cmb'])

    prefactor_kappa = probe_weights.kappa_prefactor_nosim(cosmo=astropy_cosmo)

    for s in LOGGER.progressbar(range(len(shell_info)), desc=f'bin={j+1}/{n_redshift_bins} probe={probe}'):

        if check_zbound(shell_info, s, z_lim_up, probe):

            w_k_cmb = weight_generator_kg(z_low=shell_info['lower_z'][s], 
                                          z_up=shell_info['upper_z'][s], 
                                          cosmo=astropy_cosmo)

            nz_info[j]['w_k_cmb'][s] = w_k_cmb * prefactor_kappa

    return nz_info



probe_kernel_funcs = {'kg': get_kernel_kg,
                      'dg': get_kernel_dg,
                      'ia': get_kernel_ia,
                      'k_cmb': get_kernel_k_cmb}


if __name__=='__main__':

    next(main(indices=[0], args=sys.argv[1:]))

# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Main app for calculating full sky power spectra from the CosmoGrid data and comparing it to the theory predictions with PyCosmo.

Created July 2022
author: Tomasz Kacprzak
"""

import io, os, sys, warnings, argparse, h5py, numpy as np, time, logging, itertools, shutil
from cosmogrid_des_y3 import utils_io, utils_logging, utils_config, utils_cosmogrid
from cosmogrid_des_y3.filenames import *
import healpy as hp

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
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    parser.add_argument('--write_patch_maps', action='store_true',
                        help='if to write patch maps - use only for testing')
    parser.add_argument('--par_ids', type=str, required=True,
                        help='ids of parameters to run on')

    args = parser.parse_args(args)

    utils_logging.set_all_loggers_level(args.verbosity)

    # get absolute paths
    args.config = utils_io.get_abs_path(args.config)
    args.dir_out = utils_io.get_abs_path(args.dir_out)


    return args


def resources(args):
    
    res = {'main_memory':8000,
           'main_time_per_index':4, # hours
           'main_scratch':6500,
           'merge_memory':64000,
           'merge_time':24}
           # 'pass':{'constraint': 'knl', 'account': 'des', 'qos': 'regular'}} # Cori
    
    return res


def arr_row_str(a):

    return ' '.join(f'{k}={a[k]}' for k in a.dtype.names)

def load_map(filepath_maps, probe, sample):

    with h5py.File(filepath_maps, 'r') as f:
        m = np.array(f[probe][sample])

    LOGGER.debug(f'loaded probe={probe} sample={sample} num_pix={m.shape}')
    return m

def get_patches_indices(n_patches, nside_out):

    # patches configuration
    patch_ind = hp.ud_grade(map_in=np.arange(n_patches), 
                            nside_out=nside_out, 
                            order_in='RING', 
                            order_out='RING')
    return patch_ind



def main(indices, args):
    """
    """

    args = setup(args)
    conf = utils_config.load_config(args.config)
    conf_analysis = conf['mock_analysis']
    probes = conf_analysis['probes']
    samples = np.sort([b['name'] for b in conf['paths']['redshifts_nz']])
    n_patches = patch_nside = hp.nside2npix(conf_analysis['area_frac_nside'])
    if args.test:
        LOGGER.warning('==========> TEST!')
        n_patches = 1
    
    # list of sims
    simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_simulations_list(set_type='all')
    par_ids = utils_cosmogrid.parse_par_ids(args.par_ids)
    parslist_use = parslist_all[par_ids]

    # iterate over a single map, each will be split into a number of patches
    for index in indices: 

        index_par = index%len(parslist_use)
        index_perm = index//len(parslist_use)
        params_current = parslist_use[index_par]

        LOGGER.info(f"====================================> index={index} index_par={index_par}/{len(args.par_ids)} index_perm={index_perm} path_par={params_current['path_par']}")

        if conf['projection']['shell_perms'] == False:
            raise Exception('shell_perms==False unsupported, use permuted shells ')

        # get the directory with the map for shell permutation mode
        dir_sim = get_dirname_permuted_maps(args.dir_out, params_current, id_perm=index_perm)
        LOGGER.info(f'using permuted projected maps, dir_sim={dir_sim}')
        LOGGER.info(f"maps for variant={conf_analysis['analysis_variant']}")

        # get the maps name
        filepath_maps = get_filepath_projected_maps(dir_sim, conf_analysis['analysis_variant'])
        if not os.path.isfile(filepath_maps):
            LOGGER.error(f'file not found: {filepath_maps}, continuing ...')
            continue

        dict_out = {}

        # each probe separately
        for probe in probes:

            # loop over x-corr combinations
            indices_tri = np.vstack(np.triu_indices(len(samples))).T
            LOGGER.info(f'===========> calculating probe={probe} for {len(indices_tri)} bin combinations and {n_patches} patches')

            # get the alms decompositions
            alms_s, alms_n, nside = get_maps(probe, samples, filepath_maps, conf_analysis)
            patch_ind = get_patches_indices(patch_nside, nside)
            neighbor_ind = hp.get_all_neighbours(nside, np.arange(hp.nside2npix(nside))).T

            for (i1, i2) in LOGGER.progressbar(indices_tri):

                LOGGER.debug(f"=======> bins {samples[i1]} x {samples[i2]}")
                        
                # get full sky map of cross bins
                m_s = hp.alm2map(np.sqrt(alms_s[i1])*np.sqrt(alms_s[i2]), nside=nside)
                m_n = hp.alm2map(np.sqrt(alms_n[i1])*np.sqrt(alms_n[i2]), nside=nside)

                # split the full sky map into patchespa
                for j in range(n_patches):

                    mask_patch = patch_ind != j
                    m_s_patch = m_s.copy()
                    m_s_patch[mask_patch] = hp.UNSEEN
                    m_n_patch = m_n.copy()
                    m_n_patch[mask_patch] = hp.UNSEEN

                    if args.write_patch_maps:

                    
                        fname = get_filename_patch_maps(dir_out=dir_sim,
                                                        index_perm=index_perm,
                                                        probe=probe,
                                                        index_patch=j,
                                                        id_sample1=i1,
                                                        id_sample2=i2)
                        hp.write_map(fname, m_s_patch.astype(np.float32), fits_IDL=False, overwrite=True)
                        LOGGER.info(f'wrote {fname}')
                        break # store only one patch

                    # compute fratures
                    for feature in conf_analysis['features']:
                        
                        LOGGER.debug(f'patch={j+1}/{n_patches} feature={feature}')
                        
                        # cls for the maps
                        fx, fsy = get_features(m_s_patch, feature, nside, probe, neighbor_ind, i1, i2)
                        fx, fny = get_features(m_n_patch, feature, nside, probe, neighbor_ind, i1, i2)

                        key = f"{probe}/{feature}/patch{j:02d}"
                        kx, ksy, kny = f"{key}/fx", f"{key}/fsy", f"{key}/fny"

                        dict_out.setdefault(kx, [])
                        dict_out.setdefault(ksy, [])
                        dict_out.setdefault(kny, [])

                        dict_out[kx].append(fx)
                        dict_out[ksy].append(fsy)
                        dict_out[kny].append(fny)

            if args.write_patch_maps:
                continue

            # concat data vectors
            for i in range(n_patches): 
                for feature in conf_analysis['features']:
                    
                    key = f"{probe}/{feature}/patch{i:02d}"
                    kx, ksy, kny = f"{key}/fx", f"{key}/fsy", f"{key}/fny"

                    dict_out[kx] = np.concatenate(dict_out[kx])
                    dict_out[ksy] = np.concatenate(dict_out[ksy])
                    dict_out[kny] = np.concatenate(dict_out[kny])

        if args.write_patch_maps:  # get maps only for the first index
            break

        filepath_out = get_filepath_features(dir_sim)
        store_features(filepath_out, dict_out)
            
        yield index


def merge(indices, args):

    args = setup(args)
    conf = utils_config.load_config(args.config)
    conf_analysis = conf['mock_analysis']    

    simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_simulations_list(set_type='all')
    par_ids = utils_cosmogrid.parse_par_ids(args.par_ids)
    parslist_use = parslist_all[par_ids]
    
    for id_par in range(len(parslist_use)):

        dict_out = {}
        pars = parslist_use[id_par]
        dir_root = get_dirname_probe_weights(args.dir_out, pars)
        from glob import glob
        list_output = glob(os.path.join(dir_root, '*', 'features.h5'))
        LOGGER.info(f"=====> par={id_par} n_outputs={len(list_output)} path={pars['path_par']}")
        for fname in LOGGER.progressbar(list_output, desc='reading output files'):

            with h5py.File(fname, 'r') as f:
                
                for probe in f.keys():
                    dict_out.setdefault(probe, {})

                    for feature in f[probe].keys():
                        dict_out[probe].setdefault(feature, {'fx': [], 'fsy': [], 'fny': []})

                        for patch in f[probe][feature].keys():

                            if 'fsy' not in f[probe][feature][patch].keys():
                                continue

                            dict_out[probe][feature]['fx'].append(np.array(f[probe][feature][patch]['fx']))
                            dict_out[probe][feature]['fsy'].append(np.array(f[probe][feature][patch]['fsy']))
                            dict_out[probe][feature]['fny'].append(np.array(f[probe][feature][patch]['fny']))


        # convert to arrays
        for probe in dict_out.keys():
            for feature in dict_out[probe].keys():
                dict_out[probe][feature]['fx'] = np.array(dict_out[probe][feature]['fx'])
                dict_out[probe][feature]['fny'] = np.array(dict_out[probe][feature]['fny'])
                dict_out[probe][feature]['fsy'] = np.array(dict_out[probe][feature]['fsy'])
                LOGGER.info(f"probe={probe} feature={feature} shape={dict_out[probe][feature]['fsy'].shape}")

        # store
        filepath_merged = get_filepath_features_merged(dir_root)
        utils_io.write_to_pickle(filepath_merged, dict_out)
        




def store_features(filepath_out, dict_out):

    with h5py.File(filepath_out, 'w') as f:
        for k in dict_out:
            f.create_dataset(name=k, data=dict_out[k], compression='lzf', shuffle=True)

    LOGGER.info(f'stored {filepath_out} with {len(dict_out.keys())} datasets')

def get_features(m, feature, nside, probe, neighbor_ind, i1=None, i2=None):

    edges_to_centers = lambda x: (x[1:]+x[:-1])/2
    
    if feature == 'cls':

        bins_ell_edges = np.logspace(2,3,6)
        ell_binned = edges_to_centers(bins_ell_edges)
        
        # get cls
        cl = hp.alm2cl(hp.map2alm(m))
        ell = np.arange(len(cl))
        f_sky = np.count_nonzero(m!=hp.UNSEEN)/len(m)
        n_per_bin = np.histogram(ell, bins=bins_ell_edges)[0]
        cl_binned = np.histogram(ell, bins=bins_ell_edges, weights=cl)[0]/n_per_bin/f_sky

        x, y = ell_binned, cl_binned

    elif feature == 'peaks':

        if probe == 'dg':
            m_min, m_max = 400, 700
        
        elif probe == 'kg':
            m_min, m_max = 0, 0.05


        bins_edges = np.linspace(m_min, m_max, 6)
        m_binned = edges_to_centers(bins_edges)

        m_neighbor = m[neighbor_ind]
        select_peak = np.all(np.expand_dims(m, axis=-1) > m_neighbor, axis=1) & np.all(m_neighbor != hp.UNSEEN, axis=1) & (m != hp.UNSEEN)
        m_peak = m[select_peak]
        peaks_binned = np.histogram(m_peak, bins=bins_edges)[0]
        LOGGER.debug(f'probe={probe} {i1}x{i2} feature={feature} percentiles={np.percentile(m_peak, [5, 95])}')

        x, y = m_binned, peaks_binned


    else:

        raise Exception(f'unknown feature {feature}')

    return x, y



def get_maps(probe, samples, filepath_maps, conf_analysis):

    # output container
    alms_s = np.empty(len(samples), dtype=object)
    alms_n = np.empty(len(samples), dtype=object)

    def forward_process(m, i, probe):
        m = add_probe_noise(m, probe, conf_analysis)
        a = hp.sphtfunc.map2alm(m)
        a = add_smoothing(a, fwhm_arcmin=conf_analysis['smoothing_fwhm'][probe][i])    
        return a


    for i in LOGGER.progressbar(range(len(samples)), at_level='info', desc=f'calculating alms for {len(samples)} maps'):

        # load map and convert to alms
        m_s = load_map(filepath_maps, probe, samples[i])

        # create a pure noise maps
        m_n = m_s.copy()
        m_n[:] = m_s.mean()

        # signal
        a_s = forward_process(m_s, i, probe=probe)
        a_n = forward_process(m_n, i, probe=probe)

        alms_s[i] = a_s
        alms_n[i] = a_n

        LOGGER.debug(f"got alms for probe={probe} sample={samples[i]}")

    nside = hp.npix2nside(len(m_s))

    return alms_s, alms_n, nside


def add_smoothing(a, fwhm_arcmin):

    a_smooth = hp.smoothalm(a, fwhm=np.deg2rad(fwhm_arcmin/60.)) # smoothing in arcmin
    return a_smooth


def add_probe_noise(m, probe, conf):

    nside = hp.npix2nside(len(m))
    pix_area_arcmin_sq = (np.rad2deg(np.sqrt(hp.nside2pixarea(512)))*60)**2
    n_gal_per_pix = conf['n_gal_per_arcmin_sq']*pix_area_arcmin_sq
        
    if probe == 'kg':

        sig_pix = conf['shape_noise_sig']/np.sqrt(n_gal_per_pix)
        kappa = m - np.mean(m)
        m_obs = kappa + np.random.normal(size=len(m), scale=sig_pix)

    elif probe == 'dg':

        delta = (m-np.mean(m))/np.mean(m)
        lam_pix = n_gal_per_pix*(1 + delta*conf['linear_galaxy_bias'])
        m_obs = np.random.poisson(lam_pix).astype(np.float32)

        # scale to more managable numerical range - this is already a forward scaler
        # m_obs /= n_gal_per_pix

    else:

        raise Exception(f'unknown probe {probe}')

    return m_obs
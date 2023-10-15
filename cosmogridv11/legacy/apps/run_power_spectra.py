# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Main app for calculating full sky power spectra from the CosmoGrid data and comparing it to the theory predictions with PyCosmo.

Created July 2022
author: Tomasz Kacprzak
"""

import io, os, sys, warnings, argparse, h5py, numpy as np, time, logging, itertools, shutil
from cosmogrid_des_y3 import utils_io, utils_logging, utils_config, utils_cosmogrid, utils_maps
from cosmogrid_des_y3.filenames import *
import healpy as hp

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)
from UFalcon import probe_weights

def setup(args):

    description = 'Make maps'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
                        help='logging level')
    parser.add_argument('--config', type=str, required=True, 
                        help='configuration yaml file')
    parser.add_argument('--dir_out', type=str, required=True, 
                        help='output dir for the results')
    parser.add_argument('--par_ids', type=str, required=True,
                        help='ids of parameters to run on')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    
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

def get_simulation_cls(alm1, alm2):

    xcl = hp.sphtfunc.alm2cl(alm1, alm2)
    ell = np.arange(len(xcl))

    return ell, xcl

def get_theory_cls(cosmo_params, probe1, probe2, filepath_nz1, filepath_nz2, ell_max, z_max):
    """
    Calculate PyCosmo cls.
    """

   # "cosmology": {
   #      "h": "Dimensionless hubble parameter [1]",
   #      "omega_b": "Baryon density [1]",
   #      "omega_m": "Matter density (DM+baryons) [1]",
   #      "omega_l": "Dark energy density [1]",
   #      "flat_universe": "Assume flat universe [1]",
   #      "n": "Scalar spectral index [1]",
   #      "Tcmb": "CMB Temperature [K]",
   #      "Yp": "Helim Fraction [1]",
   #      "N_massless_nu": "Number of massless neutrino species [1]",
   #      "N_massive_nu": "Number of massive neutrino species [1]",
   #      "massive_nu_total_mass": "Total mass of massive neutrinos [eV]",
   #      "wa": "Present DE of equation of state",
   #      "w0": "Time depenedent parameter of DE equation of state",

    from contextlib import redirect_stdout
    from tempfile import NamedTemporaryFile
    import PyCosmo

    # get nzs
    nz1 = utils_maps.load_nz(filepath_nz1, z_max)
    nz2 = utils_maps.load_nz(filepath_nz2, z_max)

    tmp_dir = os.environ['TMPDIR'] if 'TMPDIR' in os.environ else os.getcwd()

    tmp_f1 = os.path.join(tmp_dir, 'tmp_nz1.txt')
    tmp_f2 = os.path.join(tmp_dir, 'tmp_nz2.txt')

    np.savetxt(tmp_f1, nz1)
    np.savetxt(tmp_f2, nz2)

    # map cosmogrid probe names to pycosmo names
    probes_labels = {'kg': 'gamma', 'dg': 'deltag'}

    # general settings
    ell = np.arange(0,ell_max+1)
    clparams = {'nz': ['custom', 'custom'],
                'z0': 1.13,
                'beta': 2.,
                'alpha': 2.,
                'probes': [probes_labels[probe1], probes_labels[probe2]],
                'perturb': 'nonlinear',
                'normalised': False,
                'bias': 1.,
                'm': 0,
                'z_grid': np.array([0.00001, z_max, 400]),
                'path2zdist': [tmp_f1, tmp_f2]
                }

    # set cosmology params
    N_massive_nu = 3.0
    LOGGER.debug('fixing w0=-1, remember to switch it back when the code is working of w0<-1!!!!')
    LOGGER.debug('building cosmo')

    with redirect_stdout(io.StringIO()):

        cosmo = PyCosmo.build("mnulcdm")
        cosmo.set(pk_norm_type='sigma8',
                  massive_nu_total_mass=cosmo_params['m_nu'],
                  N_massive_nu=N_massive_nu,
                  h=cosmo_params['H0']/100,
                  omega_m=cosmo_params['Om'],
                  omega_b=cosmo_params['Ob'],
                  n=cosmo_params['ns'],
                  pk_norm=cosmo_params['s8'],
                  w0=-1, # fixing for now
                  wa=0,
                  pk_nonlin_type='rev_halofit')

    LOGGER.debug('computing theory cls')
    with redirect_stdout(io.StringIO()):
            
        #Compute Cls
        obs = PyCosmo.Obs()

        xcl = obs.cl(ell, cosmo, clparams)
        xcl[0] = 0  # replace nan

        a=np.linspace(0.001, 1, 100)
        probe1_kernel = obs.window[0](a, cosmo)
        probe2_kernel = obs.window[1](a, cosmo)

    os.remove(tmp_f1)
    os.remove(tmp_f2)

    return ell, xcl, (a, probe1_kernel, probe2_kernel)

def get_alm_decompositions(samples_probes, filepath_maps):

    # output container
    # dict_alm = {s:{} for s in samples_probes.keys()}
    # samples = np.sort(list(samples_probes.keys()))
    dict_alm = {}

    for i in LOGGER.progressbar(range(len(samples_probes)), at_level='info', desc=f'calculating alms for {len(samples_probes)} maps'):

        # decide which map to use
        s, p, _ = samples_probes[i]
        dict_alm.setdefault(p, {})

        # load map and convert to alms
        map_ = load_map(filepath_maps, p, s)
        nside = hp.npix2nside(len(map_))
        alm_ = hp.sphtfunc.map2alm(map_, lmax=2000)
        dict_alm[p][s] = alm_
        del(map_)

        LOGGER.debug(f"got alms for probe={p} sample={s}")

    return dict_alm, nside

def main(indices, args):
    """
    """

    args = setup(args)
    conf = utils_config.load_config(args.config)
    
    # get samples and probes
    samples = np.sort([b['name'] for b in conf['paths']['redshifts_nz']])
    samples_probes = []
    LOGGER.warning('ignoring ia field for now, as need to implement in pycosmo')
    for sample in conf['paths']['redshifts_nz']:
        for probe in sample['probes']:
            if probe != 'ia':
                samples_probes.append([sample['name'], probe, sample['file']])

    simslist_all, parslist_all, shell_info_all = utils_cosmogrid.get_simulations_list(set_type='all')
    par_ids = utils_cosmogrid.parse_par_ids(args.par_ids, n_max=len(parslist_all))
    parslist_use = parslist_all[par_ids]


    for index in indices: 
    
        index_par = index%len(parslist_use)
        index_perm = index//len(parslist_use)
        params_current = parslist_use[index_par]

        LOGGER.info(f"====================================> index={index} index_par={index_par} index_perm={index_perm} path_par={params_current['path_par']}")

        if conf['projection']['shell_perms'] == False:

            # get the directory with the map for no shell-permutation mode 
            dir_sim = get_dirname_projected_maps(args.dir_out, params_current, id_run=index_perm)
            LOGGER.info(f'using single_run projected maps, dir_sim={dir_sim}')
        
        else:

            # get the directory with the map for shell permutation mode
            dir_sim = get_dirname_permuted_maps(args.dir_out, params_current, project_tag=conf['tag'], id_perm=index_perm)
            LOGGER.info(f'using permuted projected maps, dir_sim={dir_sim}')

        # loop over analysis variants
        for variant in conf['analysis_variants']:

            LOGGER.info(f'==============> maps for variant={variant}')

            # get the maps name
            filepath_maps = get_filepath_projected_maps(dir_sim, variant)
            if not os.path.isfile(filepath_maps):
                LOGGER.error(f'file not found: {filepath_maps}, continuing ...')
                continue

            # get combination of samples and probes to cross-correlate
            indices_tril = np.vstack(np.tril_indices(len(samples_probes))).T

            # get the alms decompositions
            dict_alm, nside = get_alm_decompositions(samples_probes, filepath_maps)
            LOGGER.info(f'maps nside={nside}')

            # init output container
            dict_out = {}

            # loop over x-corr combinations
            LOGGER.info(f'calculating cross power spectra for {len(indices_tril)} map combinations')
            pbar = LOGGER.progressbar(indices_tril, desc=f'running xcl', at_level='info')
            for i1, i2 in pbar:
                
                # get the index of all probes and samples to x-corr
                s1, p1, f1 = samples_probes[i1]
                s2, p2, f2 = samples_probes[i2]
                alm1 = dict_alm[p1][s1]
                alm2 = dict_alm[p2][s2]

                pbar.set_description(f"{p1}/{s1} x {p2}/{s2}")
                LOGGER.debug(f"probe1={p1} sample1={s1} probe2={p2} sample2={s2}")

                # cls for the maps
                simulation_ell, simulation_xcl = get_simulation_cls(alm1, alm2)

                # cls from theory 
                theory_ell, theory_xcl, theory_kernels = get_theory_cls(params_current, p1, p2, 
                                                                        filepath_nz1=f1, 
                                                                        filepath_nz2=f2,
                                                                        ell_max=simulation_ell.max(),
                                                                        z_max=conf['probe_kernels']['z_max'])


                # output container
                key_out = f"{p1}_{s1}__{p2}_{s2}" 
                dict_out[f"{key_out}/theory_ell"] = theory_ell
                dict_out[f"{key_out}/theory_xcl"] = theory_xcl
                dict_out[f"{key_out}/simulation_ell"] = simulation_ell
                dict_out[f"{key_out}/simulation_xcl"] = simulation_xcl
                
                # write out every time, for debugging
                filepath_out = get_filepath_power_spectra(dir_sim, variant)
                store_xcl(filepath_out, dict_out, verb=False)

            # final write
            store_xcl(filepath_out, dict_out)

        yield index

def store_xcl(filepath_out, dict_out, verb=False):

    with h5py.File(filepath_out, 'w') as f:
        for k in dict_out.keys():
            f.create_dataset(name=k, data=dict_out[k], compression='lzf', shuffle=True)

    if verb:
        LOGGER.critical(f"stored {filepath_out}")

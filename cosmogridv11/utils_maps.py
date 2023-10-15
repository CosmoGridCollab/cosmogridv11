import os, sys, shutil, stat, logging, subprocess, shlex, collections, datetime, pickle, importlib, h5py 
import numpy as np, healpy as hp
from . import utils_logging, utils_config, utils_io
from UFalcon import probe_weights
from cosmogridv1.copy_guardian import NoFileException, is_remote

LOGGER = utils_logging.get_logger(__file__)

def load_nz(filepath, z_max=None):

    nz_info = np.loadtxt(filepath)
    LOGGER.debug(f'loaded redshift distribution data {filepath}')

    if z_max is not None:
        select = nz_info[:,0] <= z_max
        LOGGER.debug(f'... selected z<{z_max} {np.count_nonzero(select)}/{len(select)}')
        nz_info = nz_info[select,:]

    return nz_info

def copy_cosmogrid_file(conf, path_sim, filename, check_existing=False, noguard=False):

    from cosmogridv1.copy_guardian import CopyGuardian, NoFileException, is_remote

    filepath = os.path.join(conf['paths']['cosmogrid_root'], path_sim, filename)
    
    if not is_remote(filepath):
        
        filepath_local = filepath

    else:
        
        filepath_local = os.path.join(os.getcwd(), os.path.basename(filepath))

        if check_existing and os.path.isfile(filepath_local):
            
            LOGGER.warning(f'check_existing=True: returning local existing file {filepath_local}, make sure it is correct')

        else:
            
            copier = CopyGuardian(n_max_connect=10, n_max_attempts_remote=100, time_between_attempts=10)
            copier(sources=filepath, destination=filepath_local, force=noguard)
        
    return filepath_local

def load_probe_weigths(filename_weight, conf):

    nz_weights = {}

    with h5py.File(filename_weight, 'r') as f:

        w_shell = np.array(f['w_shell'])

        for i, sample in enumerate(conf['redshifts_nz']):

            for probe in sample['probes']:


                if probe != 'kd':
                    nz_weights.setdefault(probe, {})
                    nz_weights[probe][sample['name']] = np.array(f[probe][sample['name']])
                else:
                    # source clustering uses lensing weights array (source-lens weights)
                    nz_weights.setdefault('kg_split', {})
                    nz_weights['kg_split'][sample['name']] = np.array(f['kg_split'][sample['name']])

    LOGGER.info(f'loading probe weigths {filename_weight}') 

    return w_shell, nz_weights

def load_shells(conf, path_sim, filename_shells, check_existing=False, noguard=False):
    """Summary
    
    Parameters
    ----------
    conf : TYPE
        Description
    path_sim : TYPE
        Description
    filename_shells : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """

    filepath_shells = os.path.join(conf['paths']['cosmogrid_root'], path_sim, filename_shells)
    filepath_shells_local = copy_cosmogrid_file(conf, path_sim, filename_shells, check_existing=check_existing, noguard=noguard)

    # from cosmogridv1.copy_guardian import CopyGuardian, NoFileException, is_remote

    # filepath_shells = os.path.join(conf['paths']['cosmogrid_root'], path_sim, filename_shells)
    # filepath_shells_local = copy_cosmogrid_file(conf, path_sim, filename_shells)

    # tmp_dir = os.environ['TMPDIR'] if 'TMPDIR' in os.environ else os.getcwd()
        # if not is_remote(filepath_shells):
    #     filepath_shells_local = filepath_shells
    # else:
    #     filepath_shells_local = os.path.join(tmp_dir, os.path.basename(filepath_shells))
    #     copier = CopyGuardian(n_max_connect=10, n_max_attempts_remote=100, time_between_attempts=10)
    #     copier(sources=filepath_shells, destination=filepath_shells_local)

    shells = load_compressed_shells(filepath_shells_local, tmp_dir=None)

    LOGGER.info(f'read shells with size {shells.shape} from {filepath_shells_local}')

    return shells

def load_compressed_shells(filepath_shells, tmp_dir=None):
    """Summary
    
    Parameters
    ----------
    filepath_shells : TYPE
        Description
    tmp_dir : None, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """

    # open container
    shells_container = np.load(filepath_shells, allow_pickle=True)

    # readout and decompression - this is slow
    shells = np.array(shells_container["shells"]).astype(np.float32)

    # sort if needed
    if 'shell_info' in shells_container:
        a_sort = np.argsort(shells_container["shell_info"]["shell_name"])
    elif 'shell_infos' in shells_container:
        a_sort = np.argsort(shells_container["shell_infos"]["shell_name"][0])
    
    if not np.all(a_sort == np.arange(len(a_sort))):
        LOGGER.info('sorting array')
        shells = shells[a_sort]

    return shells



def load_shells_uncompressed(path_sim):

    from cosmogridv1.copy_guardian import CopyGuardian, NoFileException, is_remote
    import healpy as hp

    import glob
    filelist = glob.glob(os.path.join(path_sim, 'CosmoML-shell*fits'))
    filelist = np.sort(filelist)
    LOGGER.info(f'found {len(filelist)} shells in {path_sim}')

    list_maps = []
    
    for f in filelist:
        m = hp.read_map(f).astype(np.float32)
        list_maps.append(m)
        LOGGER.info(f'loaded {os.path.basename(f)} nside={hp.npix2nside(len(m))}')
    return np.array(list_maps)

def load_v11_shells(path_sim, variant):

    def add_baryonified_pixels(shells, diff_shell_inds, diff_shell_vals):

        shape_orig = shells.shape
        shells = shells.ravel()
        shells[diff_shell_inds] += diff_shell_vals
        shells = np.reshape(shells, shape_orig)

        return shells

    with h5py.File(path_sim, 'r') as f:

        shells = np.array(f['nobaryon_shells']).astype(np.float32)

        if 'dmb' in variant:
            shells = add_baryonified_pixels(shells, np.array(f['diff_shell_inds']), np.array(f['diff_shell_vals']))

    return shells




def project_shells(shells, weight, n_side, box_size, n_particles):
    """
    :param shells:
    :param weight:
    :param n_side:
    :param box_size:
    :param n_particles:
    :return m:
    """


    prefactor_sim = probe_weights.sim_spec_prefactor(n_pix=hp.nside2npix(n_side), 
                                                     box_size=box_size, 
                                                     n_particles=n_particles)

    n_pix = shells.shape[1]
    m = np.zeros(n_pix, dtype=np.float64)

    for i in range(len(shells)):

        m += shells[i].astype(np.float64) * np.float64(weight[i]) * np.float64(prefactor_sim)

    return m.astype(np.float32)


def project_shells_source_clustering(shells, weight, weight_shell, n_side, box_size, n_particles):
    """
    :param shells:
    :param weight:
    :param weight_shell:
    :param n_side:
    :param box_size:
    :param n_particles:
    :return m:
    """


    prefactor_sim = probe_weights.sim_spec_prefactor(n_pix=hp.nside2npix(n_side), 
                                                     box_size=box_size, 
                                                     n_particles=n_particles)

    n_pix = shells.shape[1]
    weight = np.where(weight<0, 0, weight)
    assert weight.ndim == 2, f'weight.shap={weight.shape}, need to pass 2 dimensional lensing weight array'

    # sum over source shells
    # arxiv:2105.13548 equation 46, version for real space kappa (not harmonic shear)
    m = np.zeros(n_pix, dtype=np.float64)
    # m_temp = np.zeros(n_pix, dtype=np.float64)
    # lens_plane = np.zeros(n_pix, dtype=np.float64)

    shells = shells.astype(np.float64)
    shells = shells - np.mean(shells, axis=-1, keepdims=True)

    for j in LOGGER.progressbar(range(len(shells)), desc='projecting source clustering shells', at_level='debug'):

        # shells_ij = shells * shells[[j], :]

        # sum over lens shells
        for i in range(0, j):

            # si = sj if i==j else shells[i]

            # source plane
            # m_temp[:] = shells[i]

            # m_temp = shells_ij[i]
            # m_temp *= weight_shell[j] * prefactor_sim

            # lens plane
            # m_temp *= shells[j]
            # m_temp *= weight[i,j] * prefactor_sim

            # si * weight_shell is the galaxy count for this shell (si-mean(si))*weight_shell is the density contrast
            source_plane = shells[i] * weight_shell[j] * prefactor_sim
            # weight[i,j] contains the n(z) contribution
            lens_plane = shells[j] * weight[i,j] * prefactor_sim

            m += source_plane * lens_plane  
            # m += m_temp

    return m.astype(np.float32)

def move_kd_to_end_of_list(probes):

    if 'kd' in probes:
        probes.remove('kd')
        probes.append('kd')
    return probes

def project_all_probes(shells, nz_info, probe_weights, shell_weight, nside, n_particles, box_size):

    # output container
    probe_maps = {}

    # project against different kernels
    for i, nzi in enumerate(nz_info):

        sample = nzi['name']

        probes = move_kd_to_end_of_list(nzi['probes'])

        for j, probe in enumerate(probes):

            LOGGER.info(f'creating map probe={probe} sample={sample}')
            probe_maps.setdefault(probe, {})

            # main magic: project shells into maps

            try:

                if probe == 'kd':

                    if 'kg' not in probe_weights.keys():
                        raise Exception('lensing weight kg required for calculating source clustering shells')


                    m = project_shells_source_clustering(shells, 
                                       weight=probe_weights['kg_split'][sample], 
                                       weight_shell=shell_weight, 
                                       n_side=nside, 
                                       box_size=box_size, 
                                       n_particles=n_particles)
                        
                else:

                    m = project_shells(shells, 
                                       weight=probe_weights[probe][sample], 
                                       n_side=nside, 
                                       box_size=box_size, 
                                       n_particles=n_particles)

                        
                probe_maps[probe][sample] = m


            except Exception as err:

                LOGGER.error(f'failed to create map errmsg={err}')

                import pudb; pudb.set_trace();
                pass

                probe_maps[probe][sample] = None

    return probe_maps


def store_probe_maps(filepath_out, probe_maps):

    LOGGER.info(f'storing in file {filepath_out}')
    with h5py.File(filepath_out, 'w') as f:

        for probe in probe_maps.keys():
            for sample in probe_maps[probe].keys():
                dset = f'{probe}/{sample}'
                if probe_maps[probe][sample] is not None:
                    f.create_dataset(name=dset, data=probe_maps[probe][sample], shuffle=True, compression="gzip", compression_opts=4)
                    LOGGER.info(f'stored {dset}')
                else:
                    LOGGER.info(f'empty map {dset}')



def check_maps_completed(conf, filepath_out, nside):

    try:

        assert os.path.isfile(filepath_out), f'file {filepath_out} not found'
        
        with h5py.File(filepath_out, 'r') as f:

            for nzi in conf['paths']['redshifts_nz']:

                sample = nzi['name']
                for probe in nzi['probes']:

                        n_pix = len(f[probe][sample])
                        assert n_pix == hp.nside2npix(nside), f'bad n_pix={n_pix}, need {hp.nside2npix(nside)}'

    except Exception as err:

        LOGGER.error(f'check on {filepath_out} failed errmsg={str(err)}')
        return False

    return True


def add_highest_redshift_shell(probe_maps, nz_info, sim_params, parslist_all):

    for nz in nz_info:

        for probe in nz['probes']:

            if 'file_high_z_cls' in nz.keys():

                for zbin in nz['file_high_z_cls'].keys():

                    cl = load_highz_cls(nz['file_high_z_cls'][zbin])

                    # cl object has a row for each row in parslist, 2521 rows, cross match by path_par
                    select = np.nonzero(sim_params['path_par'] == parslist_all['path_par'])[0][0]

                    # get cl and nside
                    cl_current = cl[select]
                    nside = hp.npix2nside(len(probe_maps[probe][nz['name']]))

                    LOGGER.info(f'adding z>3.5 synfast map for probe {probe} bin {nz["name"]}')
                    
                    # generate a synfast map
                    m = hp.sphtfunc.synfast(cl_current, nside=nside, pixwin=True, verbose=False)

                    # add to probe map
                    probe_maps[probe][nz['name']] += m


    return probe_maps

def load_highz_cls(fname):

    with h5py.File(fname, 'r') as f:
        id_params = np.array(f['id_params'])
        deltas = np.array(f['deltas'])
        benchmarks = np.array(f['benchmarks'])
        cl_kk = np.array(f['cl_kk'])

    return cl_kk



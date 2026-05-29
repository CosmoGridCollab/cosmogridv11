import os, h5py, numpy as np, healpy as hp
from . import utils_logging, utils_io

LOGGER = utils_logging.get_logger(__file__)

def load_mask(filepath):

    with h5py.File(filepath, 'r') as f:
        pix_inds = np.array(f['pix_inds_ring'])

    return pix_inds

def load_nz(filepath, z_max=None):

    assert os.path.isfile(filepath), f'file {filepath} not found, pwd={os.getcwd()}'
    nz_info = np.loadtxt(filepath)
    LOGGER.debug(f'loaded redshift distribution data {filepath}')

    if z_max is not None:
        select = nz_info[:,0] <= z_max
        LOGGER.debug(f'... selected z<{z_max} {np.count_nonzero(select)}/{len(select)}')
        nz_info = nz_info[select,:]

    if np.any(nz_info<0):
        LOGGER.warning('Detected negative entries in n(z), setting them to zero. Make sure this is acceptable.')
        nz_info = np.clip(nz_info, a_min=0, a_max=None)

    return nz_info

def cleanup_cosmogrid_files():

    dirname_out = os.path.join(os.getcwd(), 'CosmoGrid')
    utils_io.robust_remove(dirname_out)


def copy_cosmogrid_file(conf, path_sim, filename, check_existing=False, noguard=False, store_key='root', rsync_args='', rsync_R=False):

    from cosmogridv11.copy_guardian import CopyGuardian, is_remote

    if rsync_R:
        rsync_args += ' -R '

    def prepare_lists(filename, store_key):

        if type(filename) is list:
            if type(store_key) is list:
                assert len(filename)==len(store_key), f'filenames and store_keys must have the same length {len(filename)} != {len(store_key)}'
            else:
                store_key = [store_key]*len(filename)
        else:
            filename = [filename]
            if type(store_key) is list:
                store_key = [store_key[0]]
            else:
                store_key = [store_key]

        return filename, store_key


    # function input parse
    filenames, store_keys = prepare_lists(filename, store_key)

    # prepare output
    dirname_out = os.path.join(os.getcwd(), path_sim)
    utils_io.robust_makedirs(dirname_out)

    # full paths
    # filepath_in = [os.path.join(conf['paths'][f'cosmogrid_{k}'], path_sim, f) for f,k in zip(filenames, store_keys)]
    # path_sim_ = path_sim.split('raw')[1].lstrip('/') if path_sim != '' else ''
    path_sim_ = '/'.join(path_sim.split('/')[2:]) if path_sim != '' else ''
    filepath_in = [os.path.join(conf['paths'][f'cosmogrid_{k}'], path_sim_, f) for f,k in zip(filenames, store_keys)]

    files_remote, files_local = [], []
    for fpath, fname in zip(filepath_in, filenames):


        if is_remote(fpath):
            
            fpath_local = os.path.join(dirname_out, fname)

            if check_existing and os.path.isfile(fpath_local):
            
                LOGGER.warning(f'check_existing=True: returning local existing file {fpath_local}, make sure it is correct')

            else:

                files_remote.append(fpath)

            files_local.append(fpath_local)

        else:

            if os.path.isfile(fpath):

                files_local.append(fpath)

            else:

                raise Exception(f'local file not found {fpath}')

    if len(files_remote) > 0:

        copier = CopyGuardian(n_max_connect=10, n_max_attempts_remote=1000, time_between_attempts=10)
        copier(sources=files_remote, destination=dirname_out, force=noguard, rsync_args=rsync_args)

    # if not is_remote(filepath_in):
        
    #     filepath_local = filepath_in

    # else:
        
    #     # filepath_local = os.path.join(os.getcwd(), os.path.basename(filepath))
    #     dirname_out = os.path.join(os.getcwd(), path_sim)
    #     utils_io.robust_makedirs(dirname_out)
    #     filepath_local = os.path.join(dirname_out, filename)

    #     if check_existing and os.path.isfile(filepath_local):
            
    #         LOGGER.warning(f'check_existing=True: returning local existing file {filepath_local}, make sure it is correct')

    #     else:
            
    #         copier = CopyGuardian(n_max_connect=20, n_max_attempts_remote=100, time_between_attempts=10)
    #         copier(sources=filepath_in, destination=filepath_local, force=noguard)

    if type(filename) is not list:
        files_local = files_local[0]
        
    return files_local



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

    # from cosmogridv11.copy_guardian import CopyGuardian, NoFileException, is_remote

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

    if not os.path.isfile(filepath_shells):

        raise Exception(f'no file {filepath_shells}')

    # open container

    try:

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

    except Exception as err:
        LOGGER.error(f'failed to open compressed shells as numpy compressed array, trying hdf5 err={err}')

        with h5py.File(filepath_shells, 'r') as f:
            shell_info = np.array(f['shell_info'])
            npix = hp.nside2npix(2048)
            shells = np.zeros((len(shells), hp.nside2npix(2048)), dtype=np.float32)

            for i in LOGGER.progressbar(range(len(shell_info)), desc='reading shells'):
                shells[i] = np.array(f[f'shells/{i}'])

    shells = {'particles': shells}
    
    return shells



def load_shells_uncompressed(path_sim):

    from cosmogridv11.copy_guardian import CopyGuardian, NoFileException, is_remote
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

def load_v11_shells(path_sim, variant, smoothing_arcmin=None):

    def add_baryonified_pixels(shells, diff_shell_inds, diff_shell_vals):

        shape_orig = shells.shape
        shells = shells.ravel()
        shells[diff_shell_inds] += diff_shell_vals
        shells = np.reshape(shells, shape_orig)

        return shells

    shells = []
    with h5py.File(path_sim, 'r') as f:

        # shells = np.array(f['nobaryon_shells']).astype(np.float32)
        shell_keys = list(f['nobaryon_shells'].keys())
        shell_keys = np.sort(shell_keys)

        for s in shell_keys:

            shell = np.array(f['nobaryon_shells'][s]).astype(np.float32)
            
            if 'dmb' in variant:
                diff_inds = np.array(f['diff_shell_inds'][s])
                diff_vals = np.array(f['diff_shell_vals'][s])
                shell = add_baryonified_pixels(shell, diff_inds, diff_vals)
            shells.append(shell)

    shells = np.array(shells)
    out = {'particles': shells}

    if smoothing_arcmin is not None:

        shells_smooth = np.zeros_like(shells)
        from astropy import units
        for i in LOGGER.progressbar(range(len(shells)), desc='smoothing shells', at_level='info,debug'):
            shells_smooth[i] = hp.smoothing(shells[i], sigma=smoothing_arcmin[i].to(units.rad).value)
        out['particles_smoothed'] = shells_smooth

    return out

# def load_v11_haloshells(path_sim, variant):

#     with h5py.File(path_sim, 'r') as f:

#         rho = np.array(f['rho'])
#         rhosq = np.array(f['rho_sq'])

#     shells = {'halos_rho': rho, 
#               'halos_rhosq': rhosq}

#     return shells


def store_probe_maps(filepath_out, probe_maps, probe_cells=None, survey_mask=False, mode='w'):

    if survey_mask:
        pix_ind = load_mask(survey_mask)
        LOGGER.info(f'applying survey mask with {len(pix_ind)} pixels')

    LOGGER.info(f'storing in file {filepath_out}')
    with h5py.File(filepath_out, mode) as f:

        if survey_mask:
            f.create_dataset(name='pix_ind', data=pix_ind, compression='lzf', shuffle=True)

        for probe in probe_maps.keys():
            for sample in probe_maps[probe].keys():

                dset = f'map/{probe}/{sample}'
                if probe_maps[probe][sample] is not None:
                    
                    map_ = probe_maps[probe][sample]
                
                    if survey_mask:
                        map_ = map_[pix_ind]

                    f.create_dataset(name=dset, data=map_, shuffle=True, compression="gzip", compression_opts=4)
                    LOGGER.info(f'stored {dset}')

                else:
                    LOGGER.info(f'empty map {dset}')

                if probe_cells is not None:

                    dset = f'cell/{probe}/{sample}'
                    if probe_cells[probe][sample] is not None:

                        cell_ = np.array(probe_cells[probe][sample])
                        f.create_dataset(name=dset, data=cell_, shuffle=True, compression="gzip", compression_opts=4)
                        LOGGER.info(f'stored {dset}')

                    else:
                        LOGGER.info(f'empty map {dset}')


def apply_survey_mask(m, pix_ind):

    return m[pix_ind]

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



def read_particle_shells(filename_shells, shell_id='all'):

    LOGGER.info(f"loading particle shells {filename_shells} shell_id={shell_id}")

    if os.path.basename(filename_shells).endswith('.npy'):


        if shell_id == 'all':
            
            particle_shells = np.load(filename_shells)
        else:

            particle_shells = np.atleast_2d(np.load(filename_shells, mmap_mode='r')[shell_id])

    elif os.path.basename(filename_shells).endswith('.h5'):

        with h5py.File(filename_shells, 'r') as f:

            if shell_id == 'all':

                particle_shells = np.zeros((len(particle_shells_info), hp.nside2npix(2048)), dtype=np.uint16)
                for i in LOGGER.progressbar(range(len(particle_shells_info)), at_level='info'):
                    particle_shells[i] = np.array(f[f'shells/{i:03d}'])

            else:

                particle_shells = np.atleast_2d(np.array(f[f'shells/{shell_id:03d}']))

    else:
        raise Exception(f'Shell file should be .npy or .h5 {filename_shells}. To use with .npz, call decompress_particle_shells first.')

    return particle_shells



def read_shell_data(filename):

    with h5py.File(filename) as f:

        shells = np.array(f["shell_data"])

    return shells

def read_shell_info(filename_shells):

    if os.path.basename(filename_shells).endswith('.npz'):

        with np.load(filename_shells) as f:

            shell_info = np.array(f['shell_info'])

    elif os.path.basename(filename_shells).endswith('.h5'):

        with h5py.File(filename_shells, 'r') as f:

            shell_info = np.array(f['shell_info'])

    sorting = np.argsort(shell_info['shell_id'])

    shell_info = shell_info[sorting]

    return shell_info


def decompress_particle_shells(filename_shells):

    LOGGER.info(f'extracting {filename_shells}')
    with np.load(filename_shells) as f:
        fout = f.zip.extract('shells.npy')

    return fout

def decompress_halos_files(filename_halos):
    
    import tarfile
    with tarfile.open(filename_halos, "r") as tf:
        tf.extractall()

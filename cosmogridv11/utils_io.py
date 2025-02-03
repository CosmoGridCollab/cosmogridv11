import os, sys, shutil, stat, logging, subprocess, shlex, collections, datetime, numpy as np, pickle, importlib, h5py
from . import utils_logging, utils_arrays
LOGGER = utils_logging.get_logger(__file__)


# def read_particle_shells(filename_shells):

#     with np.load(filename_shells) as f:

#         LOGGER.info("loading particle shells")
#         particle_shells = np.array(f['shells'])
#         particle_shells_info = np.array(f['shell_info'])
        

#     return particle_shells, particle_shells_info


# def store_haloshells(filename_out, haloshells, dtype=np.float32):

#     with h5py.File(filename_out, 'w') as f:
#         for k, m in haloshells.items():
#             f.create_dataset(name=k, data=haloshells[k].astype(dtype), compression='gzip', compression_opts=5, shuffle=True)

#     LOGGER.info(f'stored {filename_out} with datasets {list(haloshells.keys())}')

# def get_halo_cols_to_store(tag):

#     cols_halosprop = ['ID']
#     cols_lightcone = ['x', 'y', 'z', 'shell_id', 'halo_buffer']

#     if tag == 'v1':

#         cols_halosprop += ['profile_Mvir', 'profile_Nvir', 'profile_rvir', 'profile_cvir', 'profile_success']

#     elif tag == 'v11':

#         cols_halosprop += ['m_200c', 'r_200c', 'c_200c']

#     return cols_lightcone, cols_halosprop



# def store_profiled_halos(filename_out, halo_data, shell_data, param, tag, shell_id=0):

#     cols_lightcone, cols_halosprop = get_halo_cols_to_store(tag)

#     # to save storage, only save [x, y, z, ids..] for all halos on the lightcone
#     # the other parameters are stored onece per unique halo ID
#     # cols_base = list(set(halo_data.dtype.names)-set(cols_lightcone))
#     uval, uind, uinv = np.unique(halo_data['ID'], return_index=True, return_inverse=True)
#     halo_props = utils_arrays.rewrite(halo_data[cols_halosprop]) # use only base columns
#     halo_props = halo_props[uind] # select unique halos
#     halo_pos = utils_arrays.rewrite(halo_data[cols_lightcone]) # use only lightcone columns
#     halo_pos = utils_arrays.add_cols(halo_pos, names=['uid:u8'], data=[uinv]) # add unique id

#     mode = 'w' if shell_id==0 else 'a'

#     compression_args = dict(compression='gzip', shuffle=True, compression_opts=5)
#     with h5py.File(filename_out, mode) as f:

#         f.create_dataset(name=f'shell{shell_id:03d}/halo_props', data=halo_props, **compression_args)
#         f.create_dataset(name=f'shell{shell_id:03d}/halo_pos', data=halo_pos, **compression_args)
#         f[f'shell{shell_id:03d}/halo_props'].attrs['MinParts'] = param.sim.Nmin_per_halo

#         # store shell data
#         if 'shell_data' not in f.keys():
#             f.create_dataset(name=f'shell_data', data=shell_data, **compression_args)
    
#     LOGGER.error(f'stored {filename_out} shell_id={shell_id} num_halos={len(halo_data)} with columns {str(cols_lightcone+cols_halosprop)}')

    # halos_check = read_profiled_halos(filename_out)
    # for c in cols_lightcone+cols_halosprop:
    #     assert np.all(halo_data[c]==halos_check[c]), f'halo catalog compression failed for column {c}'

# def read_shell_data(filename):

#     with h5py.File(filename) as f:

#         shells = np.array(f["shell_data"])

#     return shells

# def read_profiled_halos(filename, shell_id='all', add_xyz_shell=True):

#     # load the halos
#     with h5py.File(filename) as f:

#         shells = np.array(f["shell_data"])

#         if shell_id == 'all':

#             stack_key = lambda n: np.concatenate([np.array(f[f"shell{i:03d}/{n}"]) for i in range(len(shells))])
#             halo_pos = stack_key('halo_pos')
#             halo_pos = stack_key('halo_props')

#         else:

#             halo_pos = np.array(f[f"shell{shell_id:03d}/halo_pos"])
#             halo_props = np.array(f[f"shell{shell_id:03d}/halo_props"])

#             # np.concatenate([np.array(f[f"shell{i:03d}/halo_pos"]) for i in range(len(shells))])
#             # halo_props = np.array(f["halo_props"]) fo
#             # halos_MinParts = f["halo_props"].attrs['MinParts']

#     halos = halo_props[halo_pos['uid']]
#     halos = utils_arrays.merge_recs([halo_pos, halos])

#     if add_xyz_shell:

#         # we project the halo coordinates onto the shell they fall in
#         norm_kPc = np.sqrt(halos['x'] ** 2 + halos['y'] ** 2 + halos['z'] ** 2)
#         shell_cov = shells["shell_com"][halos["shell_id"]]
#         shell_cov_kpc = shell_cov * 1000.
#         halos = utils_arrays.add_cols(halos, names=['x_shell:f4', 'y_shell:f4', 'z_shell:f4'])
#         halos['x_shell'] = halos['x'] * shell_cov_kpc / norm_kPc
#         halos['y_shell'] = halos['y'] * shell_cov_kpc / norm_kPc
#         halos['z_shell'] = halos['z'] * shell_cov_kpc / norm_kPc

#         LOGGER.info('read {} from {}, shell_id={}'.format(len(halos), filename,  shell_id))

#     return halos


# def decompress_params_files(filename_params):

#     import tarfile
#     with tarfile.open(filename_params, "r") as tf:
#         tf.extract(member='CosmoML.log')
#         tf.extract(member='baryonification_params.py')
#         tf.extract(member='class_processed.hdf5')

# def decompress_halos_files(filename_halos_local):
    
#     import tarfile
#     with tarfile.open(filename_halos_local, "r") as tf:
#         tf.extractall()


# def load_baryonification_params(fname):

#     from cosmogridv1 import baryonification
#     sys.path.append(os.path.join(baryonification.__path__[0], '..')) # this is for back-compat with baryonification_params.py scripts
#     import importlib.util
#     spec = importlib.util.spec_from_file_location("baryonification_params.par", fname)
#     mod = importlib.util.module_from_spec(spec)
#     sys.modules["baryonification_params.par"] = mod
#     spec.loader.exec_module(mod)
#     LOGGER.info(f'using baryonification_params={mod}')
#     return mod.par



def create_random_dir(path_root=None):

    import random
    dirname = f'tmpdir{random.getrandbits(32):08x}'
    path_root = os.getcwd() if path_root is None else path_root
    path_new = os.path.join(path_root, dirname)
    if not os.path.isdir(path_new):
        robust_makedirs(path_new)
    else:
        raise Exception(f'failed to create temp dir {path_new}, already exists')
    return path_new


def write_to_pickle(filepath, obj):

    from xopen import xopen
    with xopen(filepath, mode="wb", threads=0, compresslevel=5) as f:
        pickle.dump(obj, f)

    LOGGER.info(f'wrote {filepath}')


def read_from_pickle(filepath):

    from xopen import xopen
    with xopen(filepath, mode="rb") as f:
        obj = pickle.load(f)

    LOGGER.debug(f'read {filepath}')
    return obj


def read_yaml(filename):

    import yaml
    with open(filename, 'r') as fobj:
        d = yaml.load(fobj, Loader=yaml.FullLoader)
    LOGGER.debug('read yaml {}'.format(filename))
    return d

def write_yaml(filename, d):
        
    import yaml

    with open(filename, 'w') as f:
        stream = yaml.dump(d, default_flow_style=False, width=float("inf"))
        f.write(stream.replace('\n- ', '\n\n- '))

    LOGGER.debug('wrote yaml {}'.format(filename))


def get_abs_path(path):

    if '@' in path and ':/' in path:
        abs_path = path

    elif os.path.isabs(path):
        abs_path = path

    else:
        if 'SUBMIT_DIR' in os.environ:
            parent = os.environ['SUBMIT_DIR']
        else:
            parent = os.getcwd()

        abs_path = os.path.join(parent, path)

    return abs_path

def robust_makedirs(path):

    if is_remote(path):
        LOGGER.info('Creating remote directory {}'.format(path))
        host, path = path.split(':')
        cmd = 'ssh {} "mkdir -p {}"'.format(host, path)
        subprocess.call(shlex.split(cmd))

    elif not os.path.isdir(path):
        try:
            os.makedirs(path)
            LOGGER.info('Created directory {}'.format(path))
        except FileExistsError as err:
            LOGGER.error(f'already exists {path}')

def robust_remove(dirpath):

    if os.path.isdir(dirpath):
        LOGGER.info(f'removing {dirpath}')
        shutil.rmtree(dirpath)
    else:
        LOGGER.error(f'dir {dirpath} not found')


def ensure_permissions(path, verb=True):
    val = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    os.chmod(path, val)
    if verb:
        LOGGER.debug('Changed permissions for {} to {}'.format(path, oct(val)))


def access_remove(filepath, verb=True):
    cmd = 'chmod 000 {:s}'.format(filepath)
    if verb:
        LOGGER.info('changing permissions of file {:s} to 000'.format(filepath))
    os.system(cmd)


def access_write_remove(filepath, verb=True):
    cmd = 'chmod 555 {:s}'.format(filepath)
    if verb:
        LOGGER.info('changing permissions of file {:s} to 000'.format(filepath))
    os.system(cmd)


def access_grant(filepath, verb=True):
    cmd = 'chmod 755 {:s}'.format(filepath)
    if verb:
        LOGGER.info('changing permissions of file {:s} to 755'.format(filepath))
    os.system(cmd)

def is_remote(path):
    return '@' in path and ':/' in path


def archive_results(files_to_copy, dir_out, dir_out_archive):
    
    LOGGER.info(f'moving results to archive at {dir_out_archive}')

    from cosmogridv1.copy_guardian import CopyGuardian, NoFileException
    copier = CopyGuardian(n_max_connect=10, n_max_attempts_remote=1000, time_between_attempts=10)

    list_src = []
    for src in files_to_copy:
        src = src.replace(dir_out, dir_out+'/./')
        LOGGER.info(f'copying: {src} -> {dir_out_archive}')
        list_src.append(src)

    copier(sources=list_src, destination=dir_out_archive, rsync_args=' -R --no-p -v ')
    
    for src in files_to_copy:
        LOGGER.info(f'removing {src}')
        os.remove(src)

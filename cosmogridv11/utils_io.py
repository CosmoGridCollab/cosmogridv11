import os, sys, shutil, stat, logging, subprocess, shlex, collections, datetime, numpy as np, pickle, importlib, h5py
from . import utils_logging
LOGGER = utils_logging.get_logger(__file__)



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
    copier = CopyGuardian(n_max_connect=10, n_max_attempts_remote=100, time_between_attempts=10)

    for src in files_to_copy:
        src = src.replace(dir_out, dir_out+'/./')
        LOGGER.info(f'copying: {src} -> {dir_out_archive}')
        copier(sources=src, destination=dir_out_archive, rsync_args='-R')

        LOGGER.info(f'removing {src}')
        os.remove(src)

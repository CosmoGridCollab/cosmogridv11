# Copyright (C) 2017 ETH Zurich, Cosmology Research Group

"""
Created on Aug 23, 2017
@author: Joerg Herbel
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import distutils.dir_util
import time
import random
import datetime
import subprocess
import shlex
from . import utils_logging

logger = utils_logging.get_logger(__file__)

SEMAPHORE_DIRECTORY = os.path.expanduser('~/copy_guardian_semaphores')

if not os.path.isdir(SEMAPHORE_DIRECTORY):
    try:
        os.mkdir(SEMAPHORE_DIRECTORY)
    except OSError:
        logger.warning('Semaphore directory does not exist, but it could not be created either!')


class NoFileException(Exception):
    pass


class CopyGuardian(object):

    def __init__(self, n_max_connect, n_max_attempts_remote, time_between_attempts, use_copyfile=False):

        assert time_between_attempts>0, "time_between_attempts should be >0"
        assert n_max_connect>0, "n_max_connect should be >0"
        assert n_max_attempts_remote>0, "n_max_attempts_remote should be >0"

        self.n_max_connect = n_max_connect
        self.n_max_attempts_remote = n_max_attempts_remote
        self.time_between_attempts = time_between_attempts
        self.use_copyfile = use_copyfile

    def __call__(self, sources, destination, rsync_args='', force=False):

        # Ensure correct type
        if str(sources) == sources:
            sources = [sources]

        if str(destination) != destination:
            raise ValueError('Destination {} not supported. Multiple destinations for multiple sources not '
                             'implemented.'.format(destination))

        # Ensure that rsync and local copy behave equally
        for i, source in enumerate(sources):
            if os.path.isdir(source) and not source.endswith('/'):
                sources[i] += '/'

        if destination.endswith('/'):
            destination = destination[:-1]

        # Check if destination is remote
        if is_remote(destination):

            # Check for remote sources
            for source in sources:
                if is_remote(source):
                    raise IOError('Cannot copy remote source {} to remote destination {}'.format(source, destination))

            self._copy_remote(sources, destination, rsync_args, force)

        else:
            # Split into local and remote sources
            sources_remote = []

            for source in sources:

                # Remote source
                if is_remote(source):
                    sources_remote.append(source)

                # Local source
                else:
                    self._copy_local(source, destination)

            # Now handle remaining (remote) tasks
            if len(sources_remote) > 0:
                self._copy_remote(sources_remote, destination, rsync_args, force)

    def _copy_local(self, source, destination):

        logger.info('Copying locally: {} -> {}'.format(source, destination))

        if os.path.isdir(source):
            distutils.dir_util.copy_tree(source, destination)

        elif os.path.isdir(destination) or not self.use_copyfile:
            shutil.copy(source, destination)

        else:
            shutil.copyfile(source, destination)

    def _copy_remote(self, sources, destination, rsync_args='', force=False):

        n_attempts = 0

        sources_split = self._split_sources_by_host(sources)

        while n_attempts < self.n_max_attempts_remote:

            if force:
                logger.warning('Not waiting for sync allowance, forcing download')

            else:
                self._wait_for_allowance()

            path_semaphore = self._create_semaphore()

            copied = True
            files_exist = True

            for srcs in sources_split:
                # for src in srcs:
                    # copied &= self._call_rsync([src], destination)
                rsync_result = self._call_rsync(srcs, destination, rsync_args)

                copied &= rsync_result.returncode==0

                if "No such file or directory" in rsync_result.stderr:
                    os.remove(path_semaphore)
                    logger.debug(f'removed semaphore {path_semaphore}')
                    raise NoFileException(f'Rsync failed, no such file or directory: {rsync_result.stderr}')

            try:
                os.remove(path_semaphore)
                logger.debug(f'removed semaphore {path_semaphore}')

            except Exception as err:
                logger.warning(f'failed to remove semaphore {path_semaphore}, err={str(err)}')

            if copied:
                break

            else:
                n_attempts += 1
                time.sleep(self.time_between_attempts)
                if n_attempts*self.time_between_attempts>(5 * 60):
                    logger.warning('waiting for free semaphore for long time, n_attempts={}, time={}s'.
                                   format(n_attempts, n_attempts*self.time_between_attempts))

        else:
            raise IOError("Failed to rsync {} -> {} ".format(', '.join(sources), destination))

    def _wait_for_allowance(self):

        n_attempts=0

        while True:

            time.sleep(1 + random.random()*10)

            file_list = os.listdir(SEMAPHORE_DIRECTORY)
            file_list = list(filter(lambda filename: not filename.startswith('.'), file_list))

            n_attempts+=1
            if n_attempts % 200 == 0:
                logger.info(f'waiting for semaphore n_attempts={n_attempts}')

            if len(file_list) < self.n_max_connect:
                return

    def _create_semaphore(self):
        filename = '{}_{}'.format(os.getpid(), datetime.datetime.now()).replace(' ', '_')
        filepath = os.path.join(SEMAPHORE_DIRECTORY, filename)
        open(filepath, "w").close()
        logger.debug(f'created semaphore {filepath}')
        return filepath

    def _call_rsync(self, sources, destination, rsync_args=''):

        logger.debug('Rsyncing: {} -> {}'.format(', '.join(sources), destination))

        cmd = 'rsync -av {} {} {}'.format(rsync_args, ' '.join(sources), destination)

        logger.info(cmd)

        # try:
        #     # subprocess.check_call(shlex.split(cmd))
              # return True
        # except subprocess.CalledProcessError:
        #     return False

        process_result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)


        return process_result


    def _split_sources_by_host(self, sources):

        host_dict = {}

        for s in sources:

            if ':/' in s:

                host = s.split(':/')[0]

                if host in host_dict:
                    host_dict[host].append(s)

                else:
                    host_dict[host] = [s]

            else:

                host_dict.setdefault('local', [])
                host_dict['local'].append(s)

        
        return host_dict.values()


def is_remote(path):
    return '@' in path and ':/' in path

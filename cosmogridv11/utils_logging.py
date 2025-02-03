# Copyright (C) 2017 ETH Zurich, Cosmology Research Group

"""
Created on Jul 2021
author: Tomasz Kacprzak
"""

import os, sys, logging, tqdm, time


logging_levels = {'critical': logging.CRITICAL,
                  'error': logging.ERROR,
                  'warning': logging.WARNING,
                  'info': logging.INFO,
                  'debug': logging.DEBUG}

class Timer():

    def __init__(self):
        self.time_start = {}
        self.time_start['default'] = time.time()

    def reset(self, name='default'):
        self.time_start[name] = time.time()

    def elapsed(self, name='default'):
        return '{:4.2f} min'.format((time.time()-self.time_start[name])/60.)

    def start(self, name='default'):
        self.time_start[name] = time.time()


class Progressbar():

    def __init__(self, logger=None):
        self.logger = logger
        self.intervals = {'error': 600, 'warning': 60, 'info': 10, 'debug': 1} # sec
        self.mininterval = self.intervals[logging._levelToName[self.logger.level].lower()]

    def __call__(self, collection, at_level='debug,info,warning,error', **kw):

        lvls = [logging._nameToLevel[l.upper()] for l in at_level.split(',')]
        kw.setdefault('bar_format', '{percentage:3.0f}%|{bar:28}|   {r_bar:<40} {desc}')
        kw.setdefault('disable', self.logger.level not in lvls)
        kw.setdefault('colour', 'blue')
        kw.setdefault('mininterval', self.mininterval)
        kw.setdefault('file', sys.stdout)

        from tqdm import tqdm
        return tqdm(collection, **kw)


def printt(*args):

    s = ' '.join([str(a) for a in args])

    global PRINT_TEMP_STR_LEN
    try:
        PRINT_TEMP_STR_LEN
    except:
        PRINT_TEMP_STR_LEN=0
    print('\r'+PRINT_TEMP_STR_LEN*' '+'\r'+s, end='')
    PRINT_TEMP_STR_LEN=len(s)

def decode_process_output(process_output):
    """
    Transform stdout or stderr of a process object created with subprocess.Popen to string
    :param process_output: stdout or stderr from subprocess.Popen object
    :return: output as string
    """
    return str(process_output.decode('utf-8'))


def get_logger(filepath, logging_level=None, progressbar_color='red'):
    """
    Get logger, if logging_level is unspecified, then try using the environment variable PYTHON_LOGGER_LEVEL.
    Defaults to info.
    :param filepath: name of the file that is calling the logger, used to give it a name.
    :return: logger object
    """

    if logging_level is None:
        if 'PYTHON_LOGGER_LEVEL' in os.environ:
            logging_level = os.environ['PYTHON_LOGGER_LEVEL']
        else:
            logging_level = 'info'

    logger_name = '{:>12}'.format(os.path.basename(filepath)[:12])
    logger = logging.getLogger(logger_name)

    if len(logger.handlers) == 0:
        log_formatter = logging.Formatter(fmt="%(asctime)s %(name)0.12s %(levelname).3s   %(message)s ",  datefmt="%y-%m-%d %H:%M:%S", style='%')
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(log_formatter)
        logger.addHandler(stream_handler)
        logger.propagate = False
        set_logger_level(logger, logging_level)

    logger.progressbar = Progressbar(logger)
    logger.timer = Timer()
    logger.islevel = lambda level: logging._nameToLevel[level.upper()]==logger.level
    logger.memlog = Memlog(logger)

    return logger

def set_logger_level(logger, level):

    logger.setLevel(logging._nameToLevel[level.upper()])

def set_all_loggers_level(level):

    os.environ['PYTHON_LOGGER_LEVEL'] = level

    loggerDict = logging.root.manager.loggerDict
    for key in loggerDict:
        try:
            set_logger_level(logger=loggerDict[key], level=level)
        except Exception as err:
            pass


def memory_usage_psutil():

    import os, psutil;
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    

def memory_usage_str():

    import os, psutil;
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    return '{:8.1f} MiB'.format(mem)



class Memlog():

    def __init__(self, logger=None):

        self.logger = logger

    def __call__(self,  at_level='debug,info,warning,error'):

        lvls = [logging._nameToLevel[l.upper()] for l in at_level.split(',')]

        mem = memory_usage_str()

        if self.logger.level in lvls:
            self.logger.log(self.logger.level, 'memory usage ' + mem)















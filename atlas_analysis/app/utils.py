""" Utility functions for cli """
import os
import logging
import inspect
from datetime import datetime
from collections import OrderedDict
from functools import wraps

import voxcell

LOG_DIRECTORY = '.'


class ParameterContainer(OrderedDict):
    """ A dict class used to contain and display the parameters """
    def __repr__(self):
        """ Better printing than the normal OrderedDict """
        return ', '.join(str(key) + ':' + str(val) for key, val in self.items())

    __str__ = __repr__


def log_args(logger, handler_path=None):
    """ A decorator used to redirect logger and log arguments """
    def set_logger(f, logger_path=handler_path):

        if handler_path is None:
            logger_path = os.path.join(LOG_DIRECTORY, f.__name__ + '.log')

        @wraps(f)
        def wrapper(*args, **kw):
            logger.addHandler(logging.FileHandler(logger_path))
            param = ParameterContainer(inspect.signature(f).parameters)
            for name, arg in zip(inspect.signature(f).parameters, args):
                param[name] = arg
            for key, value in kw.items():
                param[key] = value
            date_str = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            logger.info(f'{date_str}:{f.__name__} args:[{param}]')
            f(*args, **kw)
        return wrapper
    return set_logger


def split_str(value_str, new_type, sep=','):
    """ Cannot set a multi value option in click so transform args with coma to list of something"""
    return list(map(new_type, value_str.strip().split(sep)))


def load_nrrds(file_paths):
    """ Load multiple nrrd files into a list of voxeldata"""
    return list(map(voxcell.VoxelData.load_nrrd, file_paths))


def set_verbose(logger, verbose):
    """ Set the verbose level for the cli """
    logger.setLevel((logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)])

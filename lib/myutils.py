import os
import re
import glob
import yaml
from string import Template
from datetime import timedelta
from dateutil.relativedelta import relativedelta

class Utils(object):

    def __init__(self): pass

    def replace(self, s, **defs):

        s_interp = Template(s).safe_substitute(defs)

        if s_interp != s:
            s_interp = self.replace(s_interp, **defs)

        return s_interp

def str_replace(s, **defs):
    """ Recursive variable substitution. Shell-style variables ($var, ${var})
    are recursively substituted with definitions from ``defs``.

    :param s: input string containing variables.
    :param defs: input dict (kwargs) containing variable definitions.
    :return: string with variable substitutions where defined.
    """

    s_interp = Template(s).safe_substitute(defs)

    if s_interp != s:
        s_interp = str_replace(s_interp, **defs)

    return s_interp

#------------------------------------------------------------------------------

def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    :param dct: dict onto which the merge is executed
    :param merge_dct: merge_dct merged into dct
    :return: None
    """
    for k in merge_dct:
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)):  #noqa
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

#------------------------------------------------------------------------------

def read_yaml(filename):

    with open(filename, 'r') as handle:
        buf = handle.read()

    cfg = yaml.safe_load(buf)

    defs = {k:str(v) for k, v in iter(os.environ.items())}
    defs.update({k:str(v) for k, v in iter(cfg.items()) if not isinstance(v, dict)})

    buf = str_replace(buf, **defs)

    cfg = yaml.safe_load(buf)

    return cfg

#------------------------------------------------------------------------------

def parse_duration(iso_duration):
    """Parses an ISO 8601 duration string into a datetime.timedelta instance.
    Args:
        iso_duration: an ISO 8601 duration string.
    Returns:
        a datetime.timedelta instance
    """

    factor = 1
    if iso_duration[0] == '-':
        factor = -1
        iso_duration = iso_duration[1:]

    if 'T' not in iso_duration:
        iso_duration += 'T0H'
        
    m = re.match(r'^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:.\d+)?)S)?$',
        iso_duration)
    if m is None:
        raise ValueError("invalid ISO 8601 duration string")

    days = 0
    hours = 0
    minutes = 0
    seconds = 0.0

    # Years and months are not being utilized here, as there is not enough 
    # information provided to determine which year and which month.
    # Python's time_delta class stores durations as days, seconds and
    # microseconds internally, and therefore we'd have to 
    # convert parsed years and months to specific number of days.

    if m[3]:
        days = int(m[3])
    if m[4]:
        hours = int(m[4])
    if m[5]:
        minutes = int(m[5])
    if m[6]:
        seconds = float(m[6])

    return factor * timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

def add_dt(dattim, iso_duration):
    """Parses an ISO 8601 duration string into a date/time increment and
       returns the adjusted datetime object.
    Args:
        dattim: datetime
            datetime object representing the date/time to be incremented.
        iso_duration: string
            an ISO 8601 duration string.
    Returns:
        new_date: datetime
            adjusted datetime instance
    """

    factor = 1
    if iso_duration[0] == '-':
        factor = -1
        iso_duration = iso_duration[1:]

    if 'P' not in iso_duration:
        iso_duration = 'P' + iso_duration

    if 'T' not in iso_duration:
        iso_duration += 'T0H'
        
    m = re.match(r'^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:.\d+)?)S)?$',
        iso_duration)
    if m is None:
        raise ValueError("invalid ISO 8601 duration string")

    new_date = dattim

    if m[1]:
        new_date += factor * relativedelta(years=int(m[1]))
    if m[2]:
        new_date += factor * relativedelta(months=int(m[2]))
    if m[3]:
        new_date += factor * timedelta(days=int(m[3]))
    if m[4]:
        new_date += factor * timedelta(hours=int(m[4]))
    if m[5]:
        new_date += factor * timedelta(minutes=int(m[5]))
    if m[6]:
        new_date += factor * timedelta(seconds=float(m[6]))

    return new_date

#------------------------------------------------------------------------------

def find_source(paths, target):

    """Locates the path containing the target directory or file(s)
       based on a left to right prioritization of paths encoded with the
       colon (:) separator. 
    Args:
        paths: string
            list of paths using a colon delimeter. Paths are searched from 
            left to right.
        target: string
            directory, file or glob expression.
    Returns:
        path: string|None
            first path to contain the target. None is returned if the target
            is not located in any of the paths. 
    """

    for path in paths.split(':'):

        src = glob.glob(os.path.join(path, target))
        if src:
            return path

    return None

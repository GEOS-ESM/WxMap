#! /usr/bin/env python

"""Long-term Mean Wind Speed

This script is a specialized application to calculate long-term means using
monthly mean wind speed files generated for the MERRA-2 FLUID project.

This tool uses a YAML configuration file to determine the input and output
filename templates. See config.yml .

This application has the following prerequisites:

    * python-3 or later
    * meanspeed.py module
    * workflow modules (config.py, myutils)
    * YAML configuration file

Required runtime settings at the time of this documentation were:

    * setenv PYTHONPATH /home/dao_ops/gmao_packages/Workflow/Shared
    * module load python/GEOSpyD/Ana2019.10_py3.7
"""

import os
import sys
import argparse
import datetime as dt
from netCDF4 import Dataset

import config
import myutils
import meanspeed

# Retrieve command-line arguments
# ===============================

parser = argparse.ArgumentParser(description='Derives monthly LTM files from monthly mean files')
parser.add_argument('syear', metavar='start_year', type=int,
                    help='starting year as ccyy')
parser.add_argument('eyear', metavar='end_year', type=int,
                    help='ending year as ccyy')
parser.add_argument('config', metavar='config', type=str,
                    help='configuration file (.yml)')

args = parser.parse_args()

syear = args.syear
eyear = args.eyear

# Get configuration.
# ==================

cfg = config.Config()
cfg.read(args.config)

ltm_file   = cfg['ltm_wspd']['LTM_FILE']
month_file = cfg['ltm_wspd']['MONTH_FILE']

# Get environment definitions
# ===========================

ut = myutils.Utils()

defs = { k:str(v) for k,v in iter(os.environ.items()) }
defs.update( {k:str(v) for k,v in iter(cfg.items()) if not isinstance(v,dict)} )
defs.update(cfg.get('environment',{}))

# Create monthly long-term means
# from monthly mean files
# ==============================

for month in range(1,13):

    dattim   = dt.datetime(syear, month, 1)
    out_file = dattim.strftime(ut.replace(ltm_file, **defs))
    fh_out   = meanspeed.Dataset(out_file)
    print(out_file)

    for year in range(syear,eyear+1):

        dattim  = dt.datetime(year, month, 1)
        in_file = dattim.strftime(ut.replace(month_file, **defs))
        fh_in   = Dataset(in_file,  mode='r')

        print(in_file)
        fh_out.sum(fh_in, dattim)

    fh_out.write()
    fh_out.close()

sys.exit(0)

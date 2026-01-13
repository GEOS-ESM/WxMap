#! /usr/bin/env python

import os
import re
import sys
import glob
import shutil
import argparse
import datetime as dt

from multiprocessing import Pool, cpu_count

from wxlib.player import *
from wxlib.handlers import wxmap
from myutils import read_yaml, str_replace, parse_duration

# Get command-line arguments

parser = argparse.ArgumentParser(description='Weather Plot')

parser.add_argument('datetime', metavar='datetime', type=str,
       help='ISO datetime as ccyy-mm-ddThh:mm:ss')
parser.add_argument('config', metavar='config', type=str,
       help='YAML config file')
parser.add_argument('--target', '-t', metavar='target', type=str,
       help='target (from config)', default='default')

args = parser.parse_args()

dattim = re.sub('[^0-9]', '', args.datetime+'000000')[0:14]
idate = int(dattim[0:8])
itime = int(dattim[8:14])
time_dt = dt.datetime.strptime(dattim,'%Y%m%d%H%M%S')
ref_date = dt.datetime.strptime(dattim[0:8],'%Y%m%d')

# Set up environment based on field campaign

resource = read_yaml(args.config)
plays = get_plays(resource, args.target)

# Make plots

nproc = cpu_count()
for play in plays:

    if not play:
        continue

    ntask = play.get('ntask', nproc)
    pool = Pool(ntask)
    pool.map(wxmap, iter(Player(play, time_dt=time_dt)))
    pool.close()
    pool.join()

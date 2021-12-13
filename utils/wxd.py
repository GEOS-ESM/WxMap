#! /usr/bin/env python

import os
import sys
import copy

import wxservice
import interface

from request import *
from taskmanager import *

task    = TaskManager()
request = Request(interface.parse_args(sys.argv[1:]))

wx = wxservice.WXService(request)

playlist = wx.playlist()

for play in playlist:

    for request in play:

        for r in request:

            if not os.path.isfile(r['oname']):

                cmd          = copy.copy(sys.argv[1:])
                index        = cmd.index('--start_dt')               
                start_dt     = r['time_dt']
                cmd[index+1] = start_dt.strftime('%Y%m%dT%H%M%S')
    
                cmd.append('--field '  + r['field'])
                cmd.append('--region ' + r['region'])
                cmd.append('--level '  + r['level'])
                task.spawn('wxmap.py ' + ' '.join(cmd))
                break

task.wait()

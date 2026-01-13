import os
import shlex
import subprocess

def wxmap(request):

    start_dt = request['start_dt']
    end_dt = request['end_dt']
    fcst_dt = request['fcst_dt']
    t_deltat = request['t_deltat']
    themes = request.get('themes', [])
    config = request.get('configs', [])
    stream = request['stream']
    field = request['field']
    level = request['level']
    region = request['region']
    oname = request.get('oname', None)
    options = request.get('options', None)

    command = ['wxmap.py']

    command += ['--theme '+v for v in themes if v]
    command += ['--config '+v for v in config if v]

    if stream:
        command += ['--stream'] + [stream]

    t_deltat = int(round(t_deltat.total_seconds()/3600))
    command += ['--start_dt'] + [start_dt.strftime("%Y%m%dT%H%M")]
    command += ['--end_dt'] + [end_dt.strftime("%Y%m%dT%H%M")]
  # command += ['--t_deltat'] + [str(t_deltat)]
    if fcst_dt:
        command += ['--fcst_dt'] + [fcst_dt.strftime("%Y%m%dT%H%M")]

    if field:
        command += ['--field'] + [field]

    if level:
        command += ['--level'] + [str(level)]

    if region:
        command += ['--region'] + [region]

    if oname:
      # command += ['--oname'] + ['"\''+oname+'\'"']
        command += ['--oname'] + ['\''+oname+'\'']
      # command += ['--oname'] + [oname]

    if options:
        command += [options]

    command = ' '.join(command)
  # print(command)
    subprocess.call(command, shell=True, executable='/bin/bash')

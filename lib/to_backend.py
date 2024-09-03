import os
import sys
import six
import collections
import datetime as dt
import glob
import copy
import json
import logging

from dateutil.relativedelta import relativedelta
import flask

import mydatetime as mdt
import wxservice 
from find import find

anim_config = 'bottom_controls\=startstop, step, refresh, looprock,'
anim_config += 'speed, capture \\n bottom_controls_style\=padding-top:'
anim_config += '20px;padding-right:5px;padding-bottom:20px;padding-left'
anim_config += ':5px; \\n speed_labels\=Slower, Faster \\n dwell\=200ms'
anim_config += ' \\n pause\=2000 \\n start_looping\=true \\n enable_smoothing=t \\n window_size=div \\n filenames\='

#grads2html={'`b3`n':'&#8323;','`b2`n':'&#8322;','`3m`0':'&micro;','`a3`n':'&sup3;',
#            '`a12`n':'&sup1;&sup2;','`a-3`n':'&#8315;&sup3;',
#            '`a18`n':'&sup1;&#8312;','`a-2`n':'&#8315;&sup2;',
#            '`a12 `n':'&sup1;&sup2; ','`a-3 `n':'&#8315;&sup3; ','`a16 `n':'&sup1;&#8310; ',
#            '`a18 `n':'&sup1;&#8312; ','`a-2 `n':'&#8315;&sup2; ','\\':'',
#            '`f13':'','$level hPa ':'','`a2`n':'&sup2;','`b4`n':'&#8324;',
#            ', Aerosol Transport':'','`a-5`n':'&#8315;&#8309;',}
grads2html={'`b3`n':'<sub>3</sub>','`b2`n':'&#8322;','`3m`0':'&micro;','`a3`n':'&sup3;',
            '`a12`n':'&sup1;&sup2;','`a-3`n':'&#8315;&sup3;',
            '`a18`n':'&sup1;&#8312;','`a-2`n':'&#8315;&sup2;',
            '`a12 `n':'&sup1;&sup2; ','`a-3 `n':'&#8315;&sup3; ','`a16 `n':'&sup1;&#8310; ',
            '`a18 `n':'&sup1;&#8312; ','`a-2 `n':'&#8315;&sup2; ','\\':'',
            '`f13':'','$level hPa ':'','`a2`n':'&sup2;','`b4`n':'&#8324;',
            ', Aerosol Transport':'','`a-5`n':'&#8315;&#8309;',}


def fields_titles(d,config):
    """ Generates title subdictionary in fields to generate
    tooltip descriptors in webpage template
    Parameters:
        d(dict):fields dictionary
        config(dict):Config dictionary from wxservice
    Returns: d(dict): updated fields dictionary
    """
    theme=config['theme_']
    wxthemes=config[theme]
    if isinstance(wxthemes,list):
        plot_themes={}
        for th in wxthemes:
            if th not in config.keys():
                th=th.split('_')[0]
            plot_themes.update(config[th]['plot'])
    elif isinstance(wxthemes,dict):
        plot_themes=wxthemes['plot']

    d['title']={}
    for k in d['all'].keys():
        label=plot_themes[k]['title']
        for g,h in grads2html.items():
            label=label.replace(g,h)
        d['title'][k]=label
    return d

def sort_dict(d):
    d_out=collections.OrderedDict()
    for k in sorted(d.keys()):
        d_out[k]=d[k]
    return d_out

def check_date_request(r):
    for k,v in six.iteritems(r):
        if 'dt' in k and isinstance(v,(dt.date,dt.datetime)):
            if hasattr(v,'hour'):
                r[k]=mdt.datetime(v.year,v.month,v.day,v.hour,v.minute,v.second)
            else:
                r[k]=mdt.datetime(v.year,v.month,v.day,0,0,0)
    return r

def check_request(request,wx,default):
    """field,fcst_dt,tau,region,level"""
    field_cfg = wx.config([wx.config['theme_'],'plot'])
    region_cfg = wx.config['region'].keys()
    theme_cfg={'field':field_cfg,'region':list(region_cfg)}    
    stream=request.get('stream',default.stream)
    field=request.get('field',default.request['field'])
    level=request.get('level',default.request['level'])
    region=request.get('region',default.request['region'])
    tau=request.get('tau',default.request['tau'])
    #fcst=request.get('fcst',default.request['fcst_dt'])
    if default.request['fcst_dt'] is None:
        fcst=request.get('fcst',default.request['time_dt'])
        fcst_flag=1
    else:
        fcst=request.get('fcst',default.request['fcst_dt']) 
        fcst_flag=0
    flags={}
    animate=request.get('animate',0)
    track=request.get('track',['none'])
    if stream not in default.streams:
        if stream!='GEOSCFFC':
            flags.update({'stream':'Requested stream, {}, is not available.'.format(stream)})
            stream=default.request['stream']
    if isinstance(fcst,six.string_types):
        if len(fcst) >8:
            fcst = dt.datetime.strptime(
                fcst, '%Y%m%dT%H%M%S')
        else:
            fcst = dt.datetime.strptime(fcst, '%Y%m%d')
            fcst = fcst.replace(
                        hour=0, minute=0, second=0, microsecond=0)
    if fcst < default.start:
        flags.update({'fcst':[fcst,'is not available.']})
        fcst = default.request['fcst_dt']
    elif fcst > default.end:
        flags.update({'fcst':[fcst,'is not available yet.']})
        fcst = default.request['fcst_dt']
    tau_index = fcst.hour
    
    if int(tau) not in default.tau_length[tau_index]:
        if int(tau)==0: tau='01'
        else:
            flags.update({'tau':'Requested {{}}, {}, is not available.'.format(tau)})
            tau=default.request['tau'] 
    time_dt = fcst + (int(tau)+default.tau_adjust)*default.request['t_deltat'] #relativedelta(months=int(tau))
    start_dt = time_dt
    end_dt = time_dt
    if field not in theme_cfg['field']: 
        flags.update({'field':'Requested field, {}, is not available.'.format(field)})
        field=default.request['field']
        levels =  theme_cfg['field'][default.request['field']]['levels']
    else:
        levels=theme_cfg['field'][field]['levels']
    if len(levels)==1: level=levels[0]
    #elif str(level).decode('utf-8') not in [str(l).decode("utf-8") for l in levels]:
    elif str(level) not in [str(l) for l in levels]:
        if int(level)!=0:
            flags.update({'level':'Requested level, {}, is not available.'.format(level)})
        if 500 in levels: level=500
        else: level=max(levels)
    if region not in theme_cfg['region']: 
        if region!='':
            flags.update({'region':'Requested region, {}, is not available.'.format(region)})
        region=default.request['region']
    r=copy.deepcopy(default.request)
    if fcst_flag: fcst = None
    r.update({'stream':stream,'fcst_dt':fcst,'time_dt':time_dt,
                'start_dt':start_dt,'end_dt':end_dt,
                'level':str(level),'field':field,'tau':tau,'region':region,
                'animate':animate,
                'track':track,
            })
    #r=check_date_request(r)
    return r,flags

def animate_config(wx,fconfig,animate,anim_config):
    import multiprocessing
    pool = multiprocessing.Pool()
    reqs=[]
    img_list=[]
    movie_list=[]

    playlist=wx.playlist()
    for play in playlist:
        for re in play:
            for req in re:
                req['start_dt'] = req['time_dt']
                req['end_dt'] = req['time_dt']
                name = req.get_name() + '.png'
                name=req['oname']
                if not os.path.isfile(name):
                    reqs.append(req)
                #img = '/'.join([fconfig.dotstatic,name.split('static')[1]])
                img = flask.url_for('.static', filename=name.split('static/')[1])#'plots/'+name)
                movie_list.append(name)
                img_list.append(img)
    results = pool.imap_unordered(generate, reqs)
    pool.close()
    pool.join()

    # javascript config for HAniS animation
    anim_config += ', '.join(img_list)
    # if movie:
    if animate > 1:
        # check for cached movie
        movie = os.path.basename(movie_list[0])[:-4] + '.mp4'
        cached_movie = os.path.isfile(os.path.join(os.path.dirname(movie_list[0]), movie))
        if not cached_movie:
            # create movie
            flist = os.path.join(os.path.dirname(movie_list[0]), movie[:-4]+'.txt')
            with open(flist, 'a') as a:
                for m in movie_list:
                   a.write('file ' + os.path.basename(m) + ' duration 0.01\n')

            os.chdir(os.path.dirname(movie_list[0]))

            import taskmanager
            task = taskmanager.TaskManager()
            command = '/usr/local/bin/ffmpeg -f concat -r 20 -i {flist} -qscale 10 {output}'.format(flist=flist, output=movie)
            task.spawn(command)
            task.wait()

            os.chmod(flist, 0o755)
            os.remove(flist)
            os.chdir(fconfig.here)
    filename=movie_list[0].split('static')[1][:-4]+'.mp4'
    return filename,anim_config

def chmod(files,remove=False):
    for f in files:
        try:
            os.chmod(f,0o775)
            if remove:
                os.remove(f)
        except Exception:
            pass

def cleaner(file_type,output_dir):
    if file_type == 'files':
        for ext in ['.png','.mp4','.nav']:
            chmod(find(output_dir,ext=ext),remove=True)
    elif file_type == 'logger':
        if os.path.exists(log_f):
            with open(log_f, 'r+') as lf:
                lines = len(lf.readlines())
            if lines > 500:
                print('CLEARING FLUID LOG')
                lf.truncate(0)
            else:
                print('LOG INTACT, FEWER THAN 500 LINES')
    elif file_type == 'directory':
        for root, dirnames, filenames in os.walk(path, topdown=False):
            for dirname in dirnames:
                remove_empty_dir(os.path.realpath(os.path.join(root, dirname)))

def remove_empty_dir(path):
    try:
        os.rmdir(path)
    except OSError:
        pass

def make_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            os.chmod(path, 0o755)
            return 'successful'
        except OSError:
            print('Error making directory: ' + path)
            flask.abort(404)

def add_nav(wx,r,stations=None):
    try:
        # nav files for hotspots
        if 'uuid' in r.get('oname','uuid'):
            for play in wx.playlist():
                for req in play:
                    r = next(iter(req))
        nav = r['oname'][:-3] + 'nav'
        if not os.path.isfile(nav):
            if 'prototype' in nav:
                nav=nav.replace('prototype/','')
            else:
                nav=nav.replace('data_services','data_services/prototype')
        if os.path.isfile(nav):
            with open(nav) as f:
                stations = json.load(f)
            for s in stations:
                link,product,station,region=s['url'].rsplit('/',3)
                s['product']=product
                s['station']=station
                s['region']=region[8:]
    except Exception as e:
        print(e)
        stations=None
    return stations

def generate(r):
    """generates figure from requests dictionary (r)"""
    import gradsdataservice as ds
    import gradsmapservice as ms
    import wxservice 
    if r['theme'][0] in ['classic_s2s_seasonal']:
        r['tau']='{:02d}'.format(int(r['tau'])-1)
    r=check_date_request(r)
    ds_ = ds.Service()
    ms_ = ms.Service()

    wx = wxservice.WXServiceLite(r)
    wx.register(dataservice=ds_, mapservice=ms_)
    playlist = wx.playlist()

    for play in playlist:
        for re in play:
            for req in re:

                try:
                    os.makedirs(os.path.dirname(req['oname']), 0o755)
                    
                except:
                    pass
                if not os.path.isfile(req['oname']):
                    m = wx.get_map(req)
                m = ''
    ds_.p.communicate()
    del ms_
    del ds_
    return

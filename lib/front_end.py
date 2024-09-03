import os
import sys
import copy
import datetime as dt
import glob
import json
import logging
from collections import OrderedDict
from argparse import Namespace
import traceback

import flask
import yaml
import numpy as np
from dateutil.relativedelta import relativedelta

import wxservice
import to_backend

interface2req=dict(animates='animate',fields='field',levels='level',
                   motifs='motif',regions='region',streams='stream',
                   taus='tau',themes='theme',times='fcst',tracks='track')

def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

class fconfig(object):
    def __init__(self,directory):
        # filesystem
        self.directory=directory
        self.here = os.path.abspath(directory)
        self.lib = os.path.join(self.here, '../lib')
        self.bin = os.path.join(self.here, 'bin')
        self.here2=os.path.dirname(self.here)
        self.instance=self.here.rsplit('/',1)[1]
        if 'data-services' in self.here:
            self.relative=self.here.split('data-services')[-1]
        else: self.relative='/services/prototype/{0}'.format(self.instance)
        self.dotstatic=os.path.join(self.relative, '../../wxmaps/static')
        self.static=os.path.join(self.relative, '../static')
        
        # output
        self.output_dir = os.path.join(self.here, '../../wxmaps/static/plots')
        self.img = os.path.join(self.output_dir, '$field/$level/$region/$uuid.png')

        # get config
        config_file='{}/{}_fconfig.yml'.format(self.here,self.instance)
        sys.path.insert(0,self.lib)
        if not os.path.isfile(os.path.join(self.here,config_file)):
            write_yml(self.instance)

        #with open(config_file,'r') as ymlfile:
            #self.products=self.ordered_load(ymlfile)
        products=self.ordered_load(config_file)
        self.products=OrderedDict([(k,v) for k,v in products.items() if 'default' not in k])
        for k,v in self.products[self.instance].items():
            setattr(self,k,v)

    def ordered_load(self,stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
        class OrderedLoader(Loader):
            pass
        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return object_pairs_hook(loader.construct_pairs(node))
        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping)
        with open(stream,'r') as f:
            out=yaml.load(f,OrderedLoader)
        return out

        
    def write_yml(self,instance):
        from compile_fconfig_yml import write_yml
        write_yml(instance)

    def get_service(self,theme=''):
        products=self.products
        theme,base_theme,url_theme,mission=self.parse_theme(theme) 
        service=copy.deepcopy(products[base_theme])
        service['tau_int']={k:v[2] for k,v in service['tau_length'].items()}
        service['tau_length']={k:np.arange(v[0],v[1]+1,v[2]) for k,v in service['tau_length'].items()}
        for k,v in service.items():
            if 'template' not in k or not v: continue
            if any([g in k for g in self.base_temps]):
                val=[self.template_prefix,v]
            elif any([g in k for g in self.nav_temps]):
                val=[self.template_prefix,self.nav_instance,v]
            else:
                val=[self.template_prefix,self.instance,v]
            service[k]='/'.join(val)
        return Namespace(theme=theme,base_theme=base_theme,
                        url_theme=url_theme,mission=mission,**service)
    def parse_theme(self,theme):
        url_theme=theme
        if url_theme=='':theme=self.theme
        base_theme=theme
        mission=None
        if isinstance(theme,list):
            url_theme='+'.join(theme)
        else: theme=[theme]
        if '+' in url_theme:
            theme = url_theme.split('+',1)
            base_theme,mission = url_theme.split('+',1)
        base_theme = [k for k in self.products.keys() if base_theme in k][0]
        return theme,base_theme,url_theme,mission

class global_latest(object):
    def __init__(self,fconfig):
        self.fconfig=fconfig
        self.theme=self.fconfig.theme
    def defaults_internal(self,theme=None,stream=None,motif=''):
        if theme is None: theme=self.theme
        #if not hasattr(self,'stream'): self.add_theme(theme)
        if stream is None: stream=self.fconfig.stream
        service=self.fconfig.get_service(theme)
        wx = wxservice.WXServiceLite({'theme':service.theme,
                                    'bin_path':f'{self.fconfig.here}/bin',
                                    #'rc': os.path.join(self.fconfig.here, 'wxmap.rc')
                                    })
        streams=wx.config.get('streams',[stream])
        if stream not in streams: stream = service.stream
        cfg = wx.config(['stream', stream])
        start_time,end_time = self.default_time(cfg,theme=service.base_theme)
        request = self.default_request(service.theme,stream,wx,end_time,motif)
        wx = wxservice.WXServiceLite(request)
        default = Namespace(wx=wx,start=start_time,end=end_time,
                    time=end_time,request=request,service=service.title,
                    tau_length=service.tau_length,stream=stream,motif=motif,
                    streams=streams,tau_adjust=service.tau_adjust,img=None)
        return default
    def defaults(self,theme=None,stream=None,motif=''):
        default= self.defaults_internal(theme,stream,motif)
        try:
            default.img = self.latest(theme=theme,stream=stream,motif=motif)
        except Exception:
            pass 
        return default
    def global_default_request(self,theme, stream, wx, time, motif=''):
        r = {
            #'rc':os.path.join(self.fconfig.here, 'wxmap.rc'),
            'rc':os.path.join(self.fconfig.lib, 'wxmap.rc'),
            'bin_path':os.path.join(self.fconfig.here,'bin'),
            'theme':theme,
            #'motif':motif,
            'geometry':'1024x768',
            'region':wx.config['default']['region'],
            'field':wx.config['default']['field'],
            'level':wx.config['default']['level'],
            'tau': '000',
            'stream':stream,
            'collection':'',
            'time_dt':time,
            'start_dt':time,
            'end_dt':time,
            'fcst_dt':time,
            't_deltat':dt.timedelta(hours=3),
            'oname':self.fconfig.img,
            'no_title':False,
            'no_label':False,
            'plot_only':False,
            'config':[],
            'no_logo':True,
            'lights_off':False,
        }
        return r
    def default_request(self,theme,stream,wx,time,motif=''):
        r=self.global_default_request(theme,stream,wx,time,motif) 
        return r 
    def default_time(self,cfg, strftime='%Y%m%d_%H', delta=12,theme=None):
        if theme is None:
            theme=self.get_theme(type='base_theme')
        #get most recent forecast - beginning (need to benchmark)
        time = dt.date.today()
        time = dt.datetime(time.year, time.month, time.day, 12,0,0,0)
    
        #find latest file with extension = date
        from find import find
        path = os.path.dirname(os.path.dirname(cfg['uri']))
        files = [x for x in find(path=path, ext=time.strftime(strftime))]
    
        #need to limit if mount is down
        i = 10
        while not files and i > 0:
            time -= dt.timedelta(hours=delta)
            files = [x for x in find(path=path, ext=time.strftime(strftime))]
            i -= 1
    
        start_time=time-dt.timedelta(days=16)
        return start_time,time
    def global_latest(self,theme=None,stream=None,motif=''):
        
        os.chdir(self.fconfig.here)
        if theme is None: theme=self.theme
        if stream is None: stream=self.stream
        default=self.defaults_internal(theme,stream,motif)
        return self.get_latest(default.wx)
    def get_latest(self,wx):
        
        try:
            # default request
            playlist = wx.playlist()
            for play in playlist:
                for req in play:
                    r = next(iter(req))
            img = r.get_name() + '.png'
        except Exception as err:
            print(err)
            print('NO PLAYLIST: LATEST!!!!')
            flask.abort(404)
        # caching
        try:
            os.makedirs(os.path.dirname(r['oname']), 0o755)
        except:
            pass
        try:
            if not os.path.isfile(r['oname']):
                to_backend.generate(r)
            
            img=r['oname']
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            img=os.path.join(self.fconfig.static,'img/BlankMap-World6-Equirectangular.png')
        return img
    def latest(self,theme=None,stream=None,motif='',flaskargs={}):
        theme=flaskargs.get('theme',theme)
        stream=flaskargs.get('stream',stream)
        motif=flaskargs.get('motif',motif)
        img=self.global_latest(theme,stream,motif)
        return img 
 
class views(object):
    def __init__(self,fconfig): 
        self.fconfig=fconfig
        self.theme=self.fconfig.theme
        self.stream=self.fconfig.stream
        self.default_api=dict(theme=self.theme,stream=self.stream,
            motif=fconfig.motif,product=fconfig.product,
            station=fconfig.station,mission=fconfig.mission,
            pop=fconfig.pop,
            animate=0,inst='',base_url='')

    def diff_month(self,d1,d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month
        
    def default_interface(self,theme=None,stream=None,motif=''):
        if theme is None: theme=self.theme
        if stream is None: theme=self.stream
        default = self.defaults(theme,stream,motif)
        service = self.fconfig.get_service(theme)
        options = self.set_interface(default.wx,default.request,default,service,motif)
        return options

    def get_input(self,kwargs,type):
        return kwargs.get(type,self.default_api[type])

    def api_init(self,**kwargs):
        theme    = self.get_input(kwargs,'theme')
        stream   = self.get_input(kwargs,'stream')
        motif    = self.get_input(kwargs,'motif')
        default = self.defaults(theme,stream,motif)
        return theme,stream,motif,default
    
    def postx(self,**kwargs):
        flaskargs=kwargs.pop('flaskargs',{})
        kwargs.update(copy.deepcopy(flaskargs))
        theme,stream,motif,default=self.api_init(**kwargs)
        return default.img 

    def getx(self,**kwargs):
        flaskargs=kwargs.pop('flaskargs',{})
        kwargs.update(copy.deepcopy(flaskargs))
        theme,stream,motif,default=self.api_init(**kwargs)

        fconfig=self.fconfig

        stations=None
        img=None
        animate  = self.get_input(kwargs,'animate')
        if isinstance(animate,list):
            animate=animate[0]
        animate=int(animate)
        product  = self.get_input(kwargs,'product')
        station  = self.get_input(kwargs,'station')
        mission  = self.get_input(kwargs,'mission')
        pop      = self.get_input(kwargs,'pop')
        inst     = self.get_input(kwargs,'inst')
        base_url = self.get_input(kwargs,'base_url')
        service  = fconfig.get_service(theme)
        temp_vars= dict(template_prefix=fconfig.template_prefix,
                          nav_instance=fconfig.nav_instance,
                          instance=fconfig.instance,
                          inst=inst,
                          gram_function=fconfig.gram+'.api',
                          base_url=base_url.split('/')[0],
                          latest=inst+'.latest',
                          function=inst+'.api',
                          tech=inst+'.tech')
        temp_vars.update({k:v for k,v in vars(service).items() if 'template' in k})

        default = self.defaults(theme,stream,motif)
        wx=default.wx
        request,request_flags=to_backend.check_request(flaskargs,wx,default)
        
        datepicker=Namespace(time=request['time_dt'],fmt=service.datepicker_fmt,
                    end=default.end,start=default.start)
        temp_vars.update({'dp':datepicker})

        request['no_logo'] = False
        wx.request.update(request)

        opts=self.set_interface(wx,request,default,service,motif)
        anim_config = to_backend.anim_config
        error=self.get_error(request_flags,default,service)

        is_default=self.check_default_diff(default,request)
        if is_default: img=default.img; r=request;cache=True
        else:
            try:
                # non-default image
                playlist = wx.playlist()
                for play in playlist:
                    for req in play:
                        r = next(iter(req))
                        #img = r.get_name() + '.png'
                
                img=r['oname']
                cache = os.path.isfile(r['oname']) 
            except Exception as e:
                print('NO PLAYLIST!!!!')
                print(traceback.format_exc())
                print(e)
                error=Namespace(title='Data retrieval error',flag=1,
                            text=' Unable to generate requested image, default image returned.')
                img=default.img
                service.allow_anim=False
                cache=True
                opts=default_interface(theme,stream,motif)

        try:
            if animate:
                req=self.animate_request(r,service,default)
                wx_animate= wxservice.WXServiceLite(req)
                movie_file,anim_config=to_backend.animate_config(wx_animate,self.fconfig,animate,anim_config)
            
            if int(animate)>1:
                return dict(movie_file=movie_file)
        except Exception as e:
            print('NO PLAYLIST!!!!')
            print(traceback.format_exc())
            print(e)
            anim_config = to_backend.anim_config
            error=Namespace(title='Error',flag=1,
                            text=' Unable to animate map.')
 
        try:
            stations=to_backend.add_nav(wx,r)
            #print(stations) 
        except Exception as e:
            print(traceback.format_exc())
            print(e)
 
        if not cache:
            try:
                to_backend.generate(r)
            except Exception as e:
                #print(traceback.format_exc())
                #print(e)
                error.flag=1
                img=flask.url_for('static', filename='img/BlankMap-World6-Equirectangular.png')
           
        opts.update({k:v for k,v in temp_vars.items() if k!='theme'})
        
        context=dict(service=service,theme=service.url_theme,
                    anim=anim_config,animate=animate,
                    error=error,stations=stations,
                    cal_year=default.end.year,cal_month=default.end.month,
                    **opts)
        return dict(template=service.template,img=img,context=context)

    def get_error(self,request_flags,default,service):
        error=Namespace(title='500 Error',
                    text=' Requested data is unavailable at this time.',
                    flag=0)

        if request_flags:
            img = default.img
            error=Namespace(title='Input Error',flag=1,text='')
            for k,v in request_flags.items():
                if k in ['fcst']:
                    text = 'Requested {label}, {fmt}, {message}'.format(
                            label=service.fcst_label,fmt=service.fcst_fmt,message=v[1])
                    text = v[0].strftime(text)
                elif k in ['tau']:
                    text = v.format(service.tau_label)
                elif k in ['default']:
                    text = 'Unable to retrieve data to generate figure.'
                else:
                    text='{0} is not a valid option for {1}s. \n'.format(v,k)
                error.text+=text
        if 'Blank' in default.img:
            error=Namespace(title='Retrieval Error',flag=1,
                            text='Unable to retrieve data to generate maps')
        return error

    def set_arrows(self,time,dates,tau,service,time_dt,tau_fmt=['','']):
        arrow_list=['up','left','right','down']
        taus=service.tau_length[time.hour]
        try:
            itime=dates.index(time)
            time_dates=[[dates[itime+i],1] if itime+i in np.arange(len(dates)) else [time,0] for i in [-1,-1,1,1]]
            tau_dates=[time_dt,int(tau),int(tau),time_dt]
            tau_changes=[((a-b[0]).days*24+(a-b[0]).seconds/3600) if isinstance(a,dt.datetime) else a for a,b in zip(tau_dates,time_dates)]
            tau_changes=[[t,1] if t in taus else [int(tau),0] for t in tau_changes]
        except:
            time_dates =[[time,0] for i in range(4)]
            tau_changes=[[int(tau),0] for i in range(4)]                    
        try:
            itau=list(taus).index(int(tau))
            tau_leads=[[taus[t+itau],1] if t+itau in np.arange(len(taus)) and t!=0 else [int(tau),0] for t in [0,-1,1,0]]
        except:
            tau_leads=[[int(tau),0] for i in range(4)]
        time_arrows={k:{'time':ti[0].strftime(service.fcst_fmt),
                        'tau':'{0:03d}'.format(t[0]),
                        'tau_label':self.tau_fmt(ti[0],t[0],service) if ti[1]+t[1]==2 else '',
                        'time_label':ti[0].strftime(service.fcst_label_fmt) if ti[1]!=0 and t[1]!=0 else None}
                            for ti,t,k in zip(time_dates,tau_changes,arrow_list)}
    
        
        tau_arrows={k:{'time':time.strftime(service.fcst_fmt),
                       'tau':'{0:03d}'.format(t[0]),
                       'tau_label':self.tau_fmt(time,t[0],service) if t[1]!=0 else None,
                       'time_label':''}
                            for t,k in zip(tau_leads,arrow_list)}
        return time_arrows,tau_arrows

    def dud_arrows(self,time,dates,tau,service,time_dt,tau_fmt=['','']):
        arrow_list=['up','left','right','down']
        time_arrows={k:{'time':time.strftime(service.fcst_fmt),
                        'tau':'{0:03d}'.format(int(tau)),
                        'tau_label':'',
                        'time_label':None,}
                            for k in arrow_list}

        tau_arrows={k:{'time':time.strftime(service.fcst_fmt),
                        'tau':'{0:03d}'.format(int(tau)),
                        'tau_label':None,
                        'time_label':'',}
                            for k in arrow_list}
        return time_arrows,tau_arrows


    def global_interface_times(self,time,tau,service,default,request):
        if service.base_theme in ['extreme_merra2','anomaly_merra2','anomaly_merra2_seasonal','classic_s2s_seasonal']:
            type='monthly'
        else: type=''

        dates=self.global_get_dates(default,service,type)
        all_time={t.strftime(service.fcst_fmt): t.strftime(service.fcst_label_fmt) for t in dates}
        all_taus={'{:03d}'.format(t):self.tau_fmt(time,t,service) for t in service.tau_length[time.hour]}

        try:
            time_arrows,tau_arrows=self.set_arrows(time,dates,tau,service,request['time_dt'])
        except:
            time_arrows,tau_arrows=self.dud_arrows(time,dates,tau,service,request['time_dt'])
                
        times = {'selected': time.strftime(service.fcst_fmt),
                'all': to_backend.sort_dict(all_time),
                'label': service.fcst_label.upper(),
                'info':'',
                'end': default.end.strftime(service.fcst_fmt),
                'start':default.start.strftime(service.fcst_fmt),
                'arrows':time_arrows,
                }
    
        taus= {'selected': tau,
                'all': to_backend.sort_dict(all_taus),
                'label': service.tau_label.upper(),
                'info':'',
                'arrows':tau_arrows,}
    
        return times,taus

    def global_get_dates(self,default,service,type=''):
        if 'month' in type: 
            return self.global_get_mon_dates(default,service)
        
        dates = [t for t in sorted([default.start+dt.timedelta(hours=i) 
                    for i in range(0,int((default.end-default.start).days * 24+1),service.fcst_interval)])]
        return dates

    def global_get_mon_dates(self,default,service):
        dates = [t for t in [default.start+relativedelta(months=mon) for mon in range(diff_month(default.end,default.start)+1)]]
        return dates

    def global_set_interface(self,wx,request,default,service,motif):
        time=request['fcst_dt']
        if time is None:
            time=request['time_dt']
        tau=request['tau']
        response = wx.get_capabilities()
        interface = wx.get_user_interface()
    
        times,taus=self.interface_times(time,tau,service,default,request)
        streams  = {'selected': request['stream'], 'all': response['stream']}
        fields   = {'selected': request['field'], 'all': response['field']}
        levels   = {'selected': int(request['level']), 'all': response['level']}
        regions  = {'selected': request['region'], 'all': response['region']}
        animates = {'selected': request['animate'], 'all': {1:' ANIMATE',2:' DOWNLOAD MOVIE'}}
        themes   = {'selected':service.url_theme,'all':{'default':service.url_theme}}
        tracks   = {}
        motifs   = {}
        fields=to_backend.fields_titles(fields,wx.config)
        
        options= dict(title=service.title,streams=streams,fields=fields,
                    levels=levels,regions=regions,times=times,taus=taus,
                    interface=interface)
        selected={}
        for k,v in options.items():
            if 'selected' in v:
                selected.update({k:v['selected']})
        options.update({'selected':selected})
        return options

    def animate_request(self,r,service,default):
        ending_time=default.end.replace(hour=21)
        time = r['time_dt']
        r['end_dt']=min(ending_time,time+dt.timedelta(hours=60))
        r['start_dt']=max(default.start,time-dt.timedelta(hours=60))
        r['oname'] = self.fconfig.img
        return r

    def tau_fmt(self,time,t,service,order=['time','tau']):
        time_fmt=(time + dt.timedelta(hours=t)).strftime(service.tau_fmt[1])
        t_fmt=service.tau_fmt[0].format(t)
        if order[0]=='time':
            return time_fmt+t_fmt
        else:
            return t_fmt+time_fmt

    def set_interface(self,wx,request,default,service,motif):
        options=self.global_set_interface(wx,request,default,service,motif)
        return options

    def interface_times(self,time,tau,service,default,request):
        times,taus=self.global_interface_times(time,tau,service,default,request)
        return times,taus

    def check_default_diff(self,default,request):
        options=['region','time_dt','field','stream','tau','level']
        is_default=all([default.request[k]==request[k] for k in options])
        return is_default

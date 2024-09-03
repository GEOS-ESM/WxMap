import re 
from argparse import Namespace
import numpy as np 
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.colors as mc

#for font_file in font_files:
    #mpl.font_manager.fontManager.addfont(font_file)
    
class G2Py:
    def __init__(self,fconfig=None):
        self.fconfig=fconfig
        self.linestyle = {
            '1': (0,()),
            '2': (0, (14, 7)),
            '3': (0,(2,1,2,1,2,1)),
            '4': (0,(14,7,7,7)),
            '5': (0, (1, 4)),
            '6': (0,(1,4,7,4)),
            '7': (0,(1,4,1,4,7,4))}
        self.thickness={
            '1' :0.45,
            '2' :0.6,
            '3' :0.75,
            '4' :0.9375,
            '5' :1.125,
            '6' :1.3125,
            '7' :1.5,
            '8' :1.65,
            '9' :1.8,
            '10':1.95,
            '11':2.1,
            '12':2.25,
        }
        self.alignment={
            'tl':{'va':'top','ha':'left'},
            'tc':{'va':'top','ha':'center'},
            'tr':{'va':'top','ha':'right'},
            'l':{'va':'center','ha':'left'},
            'c':{'va':'center','ha':'center'},
            'r':{'va':'center','ha':'right'},
            'bl':{'va':'bottom','ha':'left'},
            'bc':{'va':'bottom','ha':'center'},
            'br':{'va':'bottom','ha':'right'},
          }
        self.font={
            '13':'Helvetica',
            '14':'Bitstream Vera Sans Mono',
            '15':'Helvetica'}
        self.rgb={
            '1': '  0   0   0',
            '0': '255 255 255',
            '2': '250  60  60',
            '3': '  0 220   0',
            '4': ' 30  60 255',
            '5': '  0 200 200',
            '6': '240   0 130',
            '7': '230 220  50',
            '8': '240 130  40',
            '9': '160   0 200',
            '10': '160 230  50',
            '11': '  0 160 255',
            '12': '230 175  45',
            '13': '  0 210 140',
            '14': '130   0 220',
            '15': '170 170 170'}
        self.strmden={
            '1': 4,#2.5/6.3,
            '2': 2.7,
            '3': 2.8,
            '4': 2.9,
            '5': 3,
            '6': 3.5,
            '7': 4,
            '8': 4.5,
            '9': 5,
            '10': 5.5,
        }
        self.marks={
            '1': dict(marker='+',fill=False),
            '2': dict(marker='o',fill=False),
            '3': dict(marker='o',fill=True),
            '4': dict(marker='s',fill=False),
            '5': dict(marker='s',fill=True),
            '6': dict(marker='x',fill=False),
            '7': dict(marker='d',fill=False),
            '8': dict(marker='^',fill=False),
            '9': dict(marker='^',fill=True),
            '10': dict(marker='o',fill=True), #Not direct GrADS conversion
            '11': dict(marker='o',fill=True), #Not direct GrADS conversion
            '12': dict(marker='d',fill=True),
        }
        
    def get_cmd(self,obj,exp):
        cmds=obj.cmds
        dfile=None
        for cmd in cmds:
            if cmd.startswith(exp):
                dfile=cmd.split(exp)[-1]
        return dfile

    def get_file_queries(self,obj):
        print(get_cmd(obj,'set dfile '))
        dfile=int(get_cmd(obj,'set dfile '))
        if dfile:
            return plot_dict['queries'][dfile]
        else:
            return None
    
    def get_grid(self,obj):
        lats=get_cmd(obj,'set lat ')
        lons=get_cmd(obj,'set lon ')
        if lats:
            lats=[float(i) for i in lats.split()]
        if lons:
            lons=[float(i) for i in lons.split()]
        return lats,lons

    def getFeature(self,feature,shape_path=''):
        fn_in=f'{shape_path}/{feature}.shp'
        shp=cartopy.io.shapereader.BasicReader(fn_in).geometries()
        shp=cartopy.feature.ShapelyFeature(shp,ccrs.PlateCarree())
        return shp

    def get_clevs(self,data,state):
        if 'clevs' in state:
            return self.array2int(state['clevs'])
        cint=float(state.get('cint',1))
        c0=np.floor(np.amin(data)/cint)*cint
        c1=np.ceil(np.amax(data)/4)*4
        cint=float(state.get('cint',1))
        cmin=float(state.get('cmin',c0))
        cmax=float(state.get('cmax',c1))

        if any([np.isnan(c) for c in [cmin,cmax]]):
            clevs = None
        else:
            clevs=np.arange(cmin,cmax+cint,cint)
        return clevs
    
    def get_color(self,color,obj=None):
        if obj:
            self.rgb.update(obj.state['rgb'])
        if len(color.split()) == 1:
            if color not in self.rgb:
                if obj is not None:
                    self.rgb.update(obj.state['rgb'])
                else:
                    color='1'
            color=self.rgb[color]
        color=self.array2int(color)/255
        return color
    
    def get_strmden(self,obj):
        strmden='5'
        for cmd in obj.cmds:
            if 'strmden' in cmd:
                strmden=[n for n in cmd.split() if n.isnumeric()][0]
                
        den=self.strmden.get(strmden)
        den = 4
        return den
    
    def parse_lines(self,obj):
        d={}
        lines=obj.state.get('line',None)
        if lines is None:
            return d
        for k,track in lines.items():
            color = track.get('line_color','31')
            lw = str(track.get('line_width','3'))
            ls=str(track.get('line_style','1'))
            color = self.get_color(color)
            lw = self.thickness[lw]
            ls=self.linestyle[ls]
            x,y=self.get_points(track['data'])
            d[k]=dict(x=x,y=y,linewidth=lw,ls=ls,color=color,linestyle='dashdot')
        return d

    def get_points(self,dat):
        points=[[float(p) for p in c.split()] for c in dat]
        x=[]
        y=[]
        for i,(a,b,c,d) in enumerate(points):
            x+=[a,c]
            y+=[b,d]
        return x,y
    
    def parse_marker(self,obj):
        for cmd in obj.cmds:
            if 'set line' in cmd:
                c = cmd.split()[2]
                continue
            elif 'draw mark' in cmd:
                t,lon,lat,s= cmd.split(' ',5)[2:]
                continue
            else:
                t = None
                c = None
        if t is None or c is None:
            return
        
        color=self.get_color(c,obj)
        m=self.marks[t]
        if not m['fill']:
            facecolor='none'
        else:
            facecolor=color
        return Namespace(x=float(lon),y=float(lat),s=s,marker=m['marker'],
                       ec=color,fc=facecolor)
    
    def parse_contour(self,obj):
        
        ccolor=obj.state.get('ccolor','31')
        ccolor=self.get_color(ccolor,obj)
        cthick=obj.state.get('cthick','3')
        cthick=self.thickness[cthick]
        fcolor,fthick,fsize=obj.state['clopts'].split()
        # fsize=100*float(fsize)        
        fsize=self.scale_fontsize(float(fsize)*2,obj.ysize/2)
        cstyle=obj.state.get('cstyle','1')
        cstyle=self.linestyle[cstyle]
        font =obj.state.get('font','13')
        font=self.font[font]
        font=self.font['13']
        copts = Namespace(ccolor=ccolor,cthick=cthick,cstyle=cstyle,
                         fcolor=fcolor,fthick=fthick,fsize=fsize,
                         font=font)
        return copts
    
    def parse_stream(self,obj):
        
        ccolor=obj.state.get('ccolor','31')
        ccolor=self.get_color(ccolor,obj)
        cthick=obj.state.get('cthick','2')
        cthick=self.thickness[cthick]
        strmden=self.get_strmden(obj)
        copts = Namespace(ccolor=ccolor,cthick=cthick,strmden=strmden,
                         )
        return copts
    
    def parse_shp(self,obj,shp,shape_path=''):
        poli = obj.state.get('poli','on')
        mpdset = obj.state.get('mpdset','mres')
        if mpdset=='hires':
            shp += ['NE_10M','LAKES']
        else: 
            shp += ['NE_50M','LAKES']
        cfeatures = []
        for s in shp:
            cfeatures.append(self.getFeature(s,shape_path))            
        mpt = obj.state.get('mpt','1 31 1 2')
        mpt = mpt.split()
        if len(mpt) < 3:
            mpt += ['1', '2']
        if len(mpt) < 4:
            mpt += ['2']
        mtype,mcolor,mstyle,mthick=mpt
        fill=[cmd.rsplit(' ',1)[-1] for cmd in obj.cmds if cmd.startswith('set shpopts')]
        if len(fill)<1 or fill[0]=='-1': fcolor=(1,1,1,0)
        else: fcolor=self.get_color(obj,fill[0])
        mcolor=self.get_color(mcolor,obj)
        mstyle=self.linestyle.get(mstyle,(0,()))
        mthick=self.thickness.get(mthick,0.75)
        shpopts=dict(edgecolor=mcolor,facecolor=fcolor,
                         linewidth=mthick)#,linestyle=mstyle)
        return cfeatures,shpopts
        
    def array2int(self,cc):
        def str2list(s):
            l=[float(ii) for ii in re.findall('[\-]*\d+\.?\d*',s)]
            return l
        if isinstance(cc,str):
            cc2=str2list(cc)
        else:
            cc2=[str2list(i) for i in cc]
        return np.array(cc2)

    def refine_rgb_list(self,cb):
        cb=self.array2int(cb)
        cb=np.divide(cb,255)
        intervals=np.linspace(0.0, 1.0, num=len(cb))
        cb_dict={}
        for i,c in enumerate(['red','green','blue','alpha']):
            cb_dict[c]=tuple(zip(intervals,cb[:,i],cb[:,i]))    

        return cb_dict

    def ccol2cbar(self,state):
        if 'clevs' not in state:
            return None
        clevs=self.array2int(state.get('clevs'))
                             
        cbar=[]
        for c in state['ccols'].split():
            cbar.append(state['rgb'][c])

        cb_dict=self.refine_rgb_list(cbar)
        cmap=mc.LinearSegmentedColormap('my_map',cb_dict,N=256)
        mapper=np.linspace(0.0, 1.0, 256)

        cm2=cmap(mapper)
        cmap2=mc.LinearSegmentedColormap.from_list('my_map', cm2, len(clevs)+1)
        norm = mc.BoundaryNorm(clevs,len(clevs)+2,extend='both')

        cbar=mpl.cm.ScalarMappable(norm=norm,cmap=cmap2)
        return cbar
    
    def check_skip(self,obj):
        ni=1
        nj=1
        for cmd in obj.cmds:
            if 'display' in cmd:
                if 'skip' in cmd:
                    skip=cmd.split('skip(')[1]
                    skip=skip.split(')')[0]
                    skips=[int(i) for i in skip.split(',') if i.isnumeric()]
                    ni=skips[0]
                    if len(skips)>1:
                        nj=skips[1]
        return ni,nj
        
    def scale_fontsize(self,fsize,ysize):
        if fsize > 5:
            size2 = fsize/1.33
        else:
            size2=fsize*100*ysize/8.5
            if ysize < 5.23:
                size2=fsize*61.5
        return size2
    def get_data(self,obj):
        if len(obj.grid)>1:
            data = Namespace(u=obj.grid[0]['data'].data,
                     v=obj.grid[1]['data'].data,
                     lon=obj.grid[0]['grid'].lon,
                     lat=obj.grid[0]['grid'].lat)
        else:
            data = Namespace(data=obj.grid[0]['data'],
                     lon=obj.grid[0]['grid'].lon,
                     lat=obj.grid[0]['grid'].lat)
        return data


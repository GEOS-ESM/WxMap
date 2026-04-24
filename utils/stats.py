#! /usr/bin/env python

import os
import sys
import copy
import math
import gridstat as gstat
import wxservice
import stat_interface
import gradsdataservice as dataservice
import gradsmapservice as mapservice

from request import *

def round_nearest(number, base):
    x = base * round(number / base)
    return int(x*10000) / 10000.0
        
def interval(range, increment):
        
    if range == 0:
        return 0
    
    x = range / float(increment)
        
    i = 1
    cint = round(x, i)
        
    while cint == 0: 
        i += 1 
        cint = round(x, i)
        
    if cint > 1:
        return round(cint)
    else:
        return cint

def set_alpha(cmin, cmax):

    alpha = []

    if cmin > 0.0:
        alpha.append("0.000 0.000 0.000")
        alpha.append("0.200 0.000 0.000")
        alpha.append("0.400 1.000 1.000")
        alpha.append("1.000 1.000 1.000")
        return alpha

    if cmax < 0.0:
        alpha.append("0.000 1.000 1.000")
        alpha.append("0.600 1.000 1.000")
        alpha.append("0.800 0.000 0.000")
        alpha.append("1.000 0.000 0.000")
        return alpha
    
    r = (0.0 - cmin) / (cmax - cmin)
    r1 = r - 0.2
    r2 = r + 0.2

    if r1 < 0.0:
        adj = -r1
        r1 += adj
        r  += adj
        r2 += adj

    if r2 > 1.0:
        adj = r2 - 1.0
        r1 -= adj
        r  -= adj
        r2 -= adj

    if r1 == 0.0:
        alpha.append("0.000 0.000 0.000")
    else:
        alpha.append("0.000 1.000 1.000")
        alpha.append("%0.3f 1.000 1.000"%(r1,))

    alpha.append("%0.3f 0.000 0.000"%(r,))

    if r2 == 1.0:
        alpha.append("1.000 1.000 1.000")
    else:
        alpha.append("%0.3f 1.000 1.000"%(r2,))
        alpha.append("1.000 1.000 1.000")

    return alpha

request = Request(interface.parse_args(sys.argv[1:]))

wx = wxservice.WXService(request)

ds = dataservice.Service()
ms = mapservice.Service()

wx.register(dataservice = ds)
wx.register(mapservice  = ms)

playlist = wx.playlist()
template = request['match'].split(':')
layer    = request['layer']

tab4=4*' '
tab6=6*' '
write_header = True

with open(request['oname'], 'w') as f:

    for play in playlist:
    
        for request in play:

            print(play)
    
            n = 0
    
            for r in request:
    
                t = r['time_dt']
                keyval = template
                token  = t.strftime(keyval[0])
                values = keyval[1].split(',')
                if write_header: f.write(tab4+r['field']+'cdict:\n')
                write_header = False
    
                for v in values:
                    if v == token:
    
                        plot   = wx.get_plot(r)[-1]
                        layers = plot.get_layer_stack('layer_names')
                      # index  = layers.index(layer)
                        index = 2
                        print(t)
                        print('===> ', layers, index)
    
                        if n == 0:
                            gs = gstat.GridStat(plot.fields[index])
                        else:
                            gs.update(plot.fields[index])
    
                        n = n + 1
    
                        break
    
            region = r['region']
            level  = str(r['level'])
            mean   = gs.mean()
            median = gs.median()
            stdmin = gs.stdmin()
            stdmax = gs.stdmax()
            rmin   = gs.amin() + stdmin * 1
            rmax   = gs.amax() - stdmax * 1
            range  = rmax - rmin

            if range == 0: continue

            cmin = rmin
            cmax = rmax
            cint = interval(cmax-cmin, 10.0)
            cmin = round_nearest(cmin, cint)
            cmax = round_nearest(cmax, cint)
    
            cmin = str(cmin); cmax = str(cmax); cint = str(cint)
    
            scale  = ''
            rfrac = (rmax - rmin) / 4.0
            if median < rmin + rfrac: scale = 'exp_scale'
            if median > rmax - rfrac: scale = 'log_scale'
    
            cfgstr = "{'$level': %s, " %(level, )
            cfgstr += "cmin: %s, cmax: %s, cint: %s"%(cmin, cmax, cint)
            if scale: cfgstr += ", scale: " + scale
            cfgstr += '}'
    
            f.write(tab6+'- '+cfgstr+'\n')

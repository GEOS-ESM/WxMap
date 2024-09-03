"""Mean Wind Speed

This module provides methods for creating a netcdf file containing mean
wind speeds. It assumes a fixed set of field names to be averaged. 

     * wspd10m
     * wspd50m
     * wspd850
     * wspd500
     * wspd700
     * wspd300
     * wspd200
"""

import os
import calendar
import collections
from netCDF4 import Dataset as dset

class Dataset(object):

    def __init__(self, fname, **kwargs):

        self.fname = fname
        self.name  = os.path.basename(fname)
        self.path  = os.path.dirname(fname)
        self.count = 0
        self.input = None

        self.stat  = {}
        self.vars  = ['wspd10m', 'wspd50m', 'wspd850',
                      'wspd500', 'wspd700', 'wspd300', 'wspd200'
                     ]

        try:
            os.makedirs(self.path, 0o755)
        except:
            pass

        self.fh = dset(fname, "w", format="NETCDF4")

        return

    def write_global_attr(self, attr, **kwargs):

        attr = collections.OrderedDict(attr)
        attr.update(kwargs.items())
        self.fh.setncatts(attr)

    def write_var(self, name, var, dims, attr, **kwargs):

        attr = collections.OrderedDict(attr)
        attr.update(kwargs.items())

        if dims and len(dims) == 1:
            self.fh.createDimension(name , var.size)

        vh = self.fh.createVariable(name, var.dtype, dims)
        vh.setncatts(attr)

        vh[:] = var

    def close(self): self.fh.close()

    def sum(self, fh, dattim):

        self.input = fh
        mdays = calendar.monthrange(dattim.year,dattim.month)[1]

        if self.count == 0:
            for name in self.vars:
                self.stat[name] = fh.variables[name][0,:,:] * 0.0

        self.count += mdays

        for name in self.vars:
            self.stat[name] += fh.variables[name][0,:,:] * mdays

    def write(self):

        if self.count == 0: return

        self.write_global_attr(self.input.__dict__,
                     Title='Long-Term Mean Wind Speed',
                     History='File written by ltm_wspd.py',
                     Filename=self.name)

        dims = ('time', 'lat', 'lon')

        time = self.input.variables['time']
        lat  = self.input.variables['lat']
        lon  = self.input.variables['lon']

        self.write_var('time',    time[:],  ('time',), time.__dict__)
        self.write_var('lat',     lat[:],   ('lat',),  lat.__dict__)
        self.write_var('lon',     lon[:],   ('lon',),  lon.__dict__)

        for name in self.vars:
            in_var = self.input.variables[name]
            self.stat[name] /= float(self.count)
            self.write_var(name, self.stat[name], dims, in_var.__dict__)

        self.count = 0

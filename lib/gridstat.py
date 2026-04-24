import math
import numpy as np
import numpy.ma as ma

class GridStat(object):

    def __init__(self, grid=None, **kwargs):

        self.bucket = {}

        self.N     = ma.count(grid)
        self.NT    = 1
        self.min   = ma.amin(grid)
        self.max   = ma.amax(grid)
        self.mmin  = ma.amin(grid)
        self.mmax  = ma.amax(grid)
        self.sum   = ma.sum(grid)
        self.sumsq = ma.sum(np.square(grid))
        print(self.min, self.max, self.mmin, self.mmax, self.sum, self.sumsq)
        self.minsum = self.min
        self.maxsum = self.max
        self.minsumsq = self.min*self.min
        self.maxsumsq = self.max*self.max

        self.cmin  = float(kwargs.get('cmin', self.min))
        self.cmax  = float(kwargs.get('cmax', self.max))

        self.bucket_sort(grid)

    def update(self, grid):

        self.N     += ma.count(grid)
        self.NT    += 1.0
        self.sum   += ma.sum(grid)
        self.sumsq += ma.sum(np.square(grid))
        mingrid     = ma.amin(grid)
        maxgrid     = ma.amax(grid)
        self.min    = min(self.min, mingrid)
        self.max    = max(self.max, maxgrid)
        self.mmin   = max(self.mmin, mingrid)
        self.mmax   = min(self.mmax, maxgrid)
        self.minsum   += mingrid
        self.maxsum   += maxgrid
        self.minsumsq += mingrid*mingrid
        self.maxsumsq += maxgrid*maxgrid

        self.bucket_sort(grid)

    def bucket_sort(self, grid):

        offset = (self.cmin + self.cmax) / 2.0
        scale  = (self.cmax - self.cmin) / 2.0

        for x in grid.compressed():
            key = int( ((x-offset)/scale) * 32767)
            nb  = self.bucket.get(key,0)
            self.bucket[key] = nb + 1

    def mean(self): return self.sum / self.N

    def amin(self): return self.min

    def amax(self): return self.max

    def ammin(self): return self.mmin

    def ammax(self): return self.mmax

    def range(self): return self.max - self.min

    def mrange(self): return self.mmax - self.mmin

    def stdmin(self):
        mean = self.minsum   / self.NT
        var  = self.minsumsq / self.NT - mean * mean
        return math.sqrt(var)

    def stdmax(self):
        mean = self.maxsum   / self.NT
        var  = self.maxsumsq / self.NT - mean * mean
        return math.sqrt(var)

    def stdev(self):
        mean = self.sum / self.N
        var  = self.sumsq / self.N - mean * mean
        return math.sqrt(var)

    def median(self):

        n      = 0
        mid    = int(round(self.N / 2.0))
        offset = (self.cmin + self.cmax) / 2.0
        scale  = (self.cmax - self.cmin) / 2.0
        median = 0

        for key in list(sorted(self.bucket.keys())):

            n += self.bucket[key]

            if n >= mid:
                median = float(key) * scale / 32767.0 + offset
                break

        return median

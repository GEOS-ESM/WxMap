#--------------------------------------------------------------------------
#
#    Copyright (C) 2006-2008 by Arlindo da Silva <dasilva@opengrads.org>
#    All Rights Reserved.
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation# using version 2 of the License.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY# without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program# if not, please consult  
#              
#              http://www.gnu.org/licenses/licenses.html
#
#    or write to the Free Software Foundation, Inc., 59 Temple Place,
#    Suite 330, Boston, MA 02111-1307 USA
#
#------------------------------------------------------------------------

""" 
This module extends the GrADS client class by providing methods for
exchanging n-dimensional NumPy array data between Python and
GrADS.
"""

__version__ = '2.0.1'
import re
import io
import six
from itertools import product
from datetime import datetime
import datetime as dt
from collections import OrderedDict
from argparse import Namespace
try:
    from mygrads.gacore       import *
    from mygrads.numtypes     import *
    from mygrads.simplekml    import SimpleKML
except Exception:
    from gacore       import *
    from numtypes     import *
    from simplekml    import SimpleKML

import numpy as np
from numpy        import zeros, ones, average, newaxis, sqrt, pi, cos, inner, \
                         arange, fromfile, float32, ma, reshape, ndarray, \
                         abs, size, meshgrid, shape, tile

from numpy.linalg import svd, lstsq

py_version=sys.version_info.major
if py_version==2: StringTypes=(str,unicode)
else: StringTypes=(str,bytes)

class GaNum(GaCore):
    """
    This class extends the GrADS client class by providing methods
    for data exchange and numerical computations:

    _Methods provided:
       exp  -  exports a GrADS expression into a NumPy array, with metada
       imp  -  imports NumPy array (+metadata) into GrADS
       eof  -  compute Empirical Orthogonal Functions (EOFS) from expressions 
       lsq  -  least square parameter estimation from expressions 

    """

#........................................................................

    def exp (self, expr, dx=None, dy=None):
        """
        Exports GrADS expression *expr*, returning a GrADS Field.

            F = self.exp(expr)

        where

            F  ---  GrADS field
            
        Generalized Expressions
        =======================

        For convenience, *expr* can also be a GrADS Field.  In such
        cases, the input Field is just returned back. This
        *overloading* is useful for writing high level functions that
        work equaly on a GrADS expression to be exported or on GrADS
        fields already known to python.

        Limitation
        ==========

        This function does not handle varying ensemble dimensions in
        GrADS v2.

        """

        missing=-9.99e8
        if py_version==3:
            arr,undef,grid=self.py3exp(expr,dx,dy)
            
            if missing in arr:
                undef=missing
            return GaField(arr,name=expr,grid=grid,mask=(arr==undef))
#       If IPC extension is not available, then try expr() instead
#       ----------------------------------------------------------
        if not self.HAS_IPC:
            return self.expr(expr)

#       For convenience, allows calls where expr is not a string, in which
#        case it returns back the input field or raise an exception
#       -------------------------------------------------------------------
        if isinstance(expr,StringTypes):
            pass # OK, will proceed to export it from GrADS
        elif isinstance(expr,GaField):
            return expr # just return input
        elif isinstance(expr,ndarray):
            return expr # this is handy for 'lsq'
        else:
            raise GrADSError("input <expr> has invalid type: %s"%type(expr))

#       Retrieve dimension environment
#       ------------------------------
        dh = self.query("dims", Quiet=True) 
        t1, t2 = dh.t
        z1, z2 = dh.z 
        nx, ny, nz, nt = (dh.nx, dh.ny, dh.nz, dh.nt)
        
#       Shortcut for 2D slices (any 2 dimensions)
#       -----------------------------------------
        if dh.rank ==2:
            return self._exp2d(expr)

#       Initial implementation: require x,y to vary for rank>2
#       Note: remove this restriction is not very hard, but requires
#             special handling the different dimension permutations separately
#             given the way GrADS invokes functions for XZ, YZ, ZT, etc
#       ----------------------------------------------------------------------
        if nx==1: raise GrADSError('lon must be varying but got nx=1')
        if ny==1: raise GrADSError('lat must be varying but got ny=1')

#       Loop over time/z, get a GrADS 2D slice at a time/z
#       --------------------------------------------------
        l = rc = 0  
        Data = None # defer allocations until we know the size
        grid = GaGrid(expr)
        grid.meta = zeros((nt,nz,20),dtype=float32)
        grid.denv = dh 
        grid.time = [] 
        grid.lev = zeros(nz,dtype=float32)         
        try:
            for t in range(t1,t2+1):
                self.cmd("set t %d"%t,Quiet=True) 
                self.cmd("q time",Quiet=True)
                grid.time.append(self.rword(1,3))
                k = 0
                for z in range(z1,z2+1):
                    self.cmd("set z %d"%z,Quiet=True)
                    field = self._exp2d(expr)
                    if Data is None:
                        ny_, nx_ = field.shape # may differ from dh.nx/dh.ny
                        Data = zeros(shape=(nt,nz,ny_,nx_), dtype=float32)
                    Data[l,k,:,:] = field.data
                    grid.lev[k] = field.grid.lev[0]
                    grid.meta[l,k,:] = field.grid.meta
                    k = k + 1
                l = l + 1

#           Record lat/lon
#           --------------
            grid.lat = field.grid.lat
            grid.lon = field.grid.lon
            amiss = grid.meta[0]

#           Remove dimensions with size 1
#           -----------------------------
            if nz==1: 
                Data = Data.reshape(nt,ny_,nx_)
                grid.dims = [ 'time', 'lat', 'lon' ]
                grid.meta = grid.meta.reshape(nt,20)
            elif nt==1: 
                Data = Data.reshape(nz,ny_,nx_)
                grid.dims = [ 'lev', 'lat', 'lon' ]
                grid.meta = grid.meta.reshape(nz,20)
            else:
                grid.dims = [ 'time', 'lev', 'lat', 'lon' ]

        except:
            self.setdim(dh)
            raise GrADSError('could not export <%s>'%expr)


        grid.tyme = array([gat2dt(t) for t in grid.time])

#       Restore dimension environment
#       -----------------------------
        self.setdim(dh)
        return GaField(Data, name=expr, grid=grid, 
                       mask=(Data==amiss), dtype=float32)
    
#........................................................................

    def _exp2d ( self, expr, dh=None ):
        """ 
        Exports GrADS expression *expr* as a GrADS Field.
        The stdio pipes are used for data exchange.
        This is an internal version handling 2D xy slices.
        In here, *expr* must be a string.
        """

        if dh==None:
            dh = self.query("dims",Quiet=True)

#       Check environmnet
#       -----------------
        nx, ny = (dh.nx, dh.ny)
        if dh.rank !=2:
            raise GrADSError('expecting rank=2 but got rank=%d'%dh.rank)

#       Create output handle, fill in some metadata
#       -------------------------------------------
        grid = GaGrid(expr)
        grid.denv = dh

#       Issue GrADS command, will check rc later
#       -----------------------------------------
        if self.Version[1]=='1':
            cmd = 'ipc_define void = ipc_save('+expr+',-)\n'
        else:
            cmd = 'define void = ipc_save('+expr+',-)\n'

        self.pywriter(cmd)

#       Position stream pointer after <EXP> marker
#       ------------------------------------------
        got = ''
        while got[:5] != '<EXP>' :
            got = self.pyreader()

#       Read header
#       -----------
        grid.meta = fromfile(self.Reader,count=20,dtype=float32)
        
        amiss = grid.meta[0]
        id = int(grid.meta[1])
        jd = int(grid.meta[2])
        nx_ = int(grid.meta[3])
        ny_ = int(grid.meta[4])

#        if id!=0 or jd!=1:
        if id<0 or id>3 or jd<0 or jd>3 or id==jd:
            self.flush()
            raise GrADSError(
                  'invalid exchange metadata (idim,jdim)=(%d,%d) - make sure <%s> is valid and that lon/lat is varying.'%(id,jd,expr))

#       Read data and coords
#       --------------------
        try:
            array_ = fromfile(self.Reader,count=nx_*ny_,dtype=float32)
            grid.lon = fromfile(self.Reader,count=nx_,dtype=float32)
            grid.lat = fromfile(self.Reader,count=ny_,dtype=float32)
        except:
            self.flush()
            raise GrADSError('problems exporting <'+expr+'>, fromfile() failed')

#       Annotate grid - assumes lon, lat
#       --------------------------------
        dims = ( 'lon', 'lat', 'lev', 'time' )
        grid.dims = [dims[jd],dims[id]]
        grid.time = [ dh.time[0] ]
        grid.lev = ones(1,dtype=float32) * float(dh.lev[0])

#       Check rc from asynchronous ipc_save
#       -----------------------------------
        rc,_ = self._parseReader(Quiet=True)
        if rc:
            self.flush()
            raise GrADSError('problems exporting <'+expr+'>, ipc_save() failed')

        grid.tyme = array([gat2dt(t) for t in grid.time])

#       Create the GaField object
#       -------------------------
        data = array_.reshape(ny_,nx_)
        self.flush()
        return GaField(data, name=expr, grid=grid, mask=(data==amiss) )
    
#........................................................................

    def imp ( self, name, Field ):
        """
        Sends a GrADS Field containing a NumPy array and associated 
        grid information to GrADS, defining it in GrADS as *name*.
        Notice that *Field* can be an instance of the GaField
        class or a tuple with the (Array,Grid) components.

        Limitation
        ==========

        This function does not handle varying ensemble dimensions in
        GrADS v2.

        """

#       If IPC extension is not available, barf
#       ---------------------------------------
        if not self.HAS_IPC:
            raise GrADSError( "IPC extension not available - cannot import!")

#       Resolve Field
#       -------------
        if isinstance(Field,GaField):
            grid = Field.grid
        else:
            raise GrADSError("Field has invalid type")
                
#       Retrieve dimension environment
#       ------------------------------
        dh = self.query("dims", Quiet=True) 
        t1, t2 = dh.t
        z1, z2 = dh.z 
        nx, ny, nz, nt = (dh.nx, dh.ny, dh.nz, dh.nt)
        nxy = nx * ny

#       Initial implementation: require x,y to vary
#       Note: remove this restriction is not very hard, but requires
#             special handling the different dimension permutations separately
#             given the way GrADS invokes functions for XZ, YZ, ZT, etc
#       ----------------------------------------------------------------------
        if nx==1: raise GrADSError('lon must be varying but got nx=1')
        if ny==1: raise GrADSError('lat must be varying but got ny=1')

#       Determine actual load command
#       -----------------------------
        if name == '<display>':
            cmd = 'display ipc_load()\n'
            if nz>1 and nt>1:
                raise GrADSError(
                      'for <display> only one of z/t can vary'+\
                      ' but got (nz,nt)=(%d,%d)'%(nz,nt) )
        else:
            if self.Version[1]=='1':
                cmd = 'ipc_define %s = ipc_load()\n'%name
            else:
                cmd = 'define %s = ipc_load()\n'%name

#       Tell GrADS to start looking for data in transfer stream
#       -------------------------------------------------------
        try:
            self.cmd("ipc_open - r")
        except GrADSError:
            raise GrADSError('<ipc_open - r> failed; is IPC installad?')
        self.pywriter(cmd) # asynchronous transfer

#       Reshape and get original t/z offset
#       -----------------------------------
        t1_, z1_ = (grid.denv.t[0], grid.denv.z[0])
        nt_, nz_ = (grid.denv.nt,grid.denv.nz)
        nx_ = len(grid.lon)
        ny_ = len(grid.lat)
        nxy_ = nx_ * ny_
        data = Field.data.reshape(nt_,nz_,ny_,nx_)
        meta = grid.meta.reshape(nt_,nz_,20)

#       Write the data to transfer stream
#       ----------------------------------
        try:
            for t in range(t1,t2+1):
                l = t - t1_
                for z in range(z1,z2+1):
                    k = z - z1_
                    mx = int(meta[l,k,3])
                    my = int(meta[l,k,4])
                    if mx!=nx_ or my!=ny_:
                        self.flush()
                        raise GrADSError(
                             'nx/ny mismatch; got (%d,%d), expected (%d,%d)'%\
                             (mx,my,nx_,ny_))
                    meta[l,k,:].tofile(self.Writer)
                    data[l,k,:,:].tofile(self.Writer)
                    grid.lon.tofile(self.Writer)
                    grid.lat.tofile(self.Writer)
                    self.Writer.flush()
        except:
            self.flush()
            self.setdim(dh)
            raise GrADSError(
                  'could not import <%s>, tofile() may have failed'%name
)

#       Check rc from asynchronous ipc_save
#       -----------------------------------
        rc = self._parseReader(Quiet=True)
        self.flush()

#       Restore dimension environment
#       -----------------------------
        self.setdim(dh)
        self.cmd("ipc_close")
        if rc:
            raise GrADSError('problems importing <'+name+'>, ipc_load() failed')

#........................................................................

    def expr (self, expr):
        """
        Evaluates a GrADS expression returning a GrADS Field. This is similar
        to the exp() method except that the resulting GaField cannot be
        imported back into GrADS. It relies on *gacore* methods eval()
        and coords() to retrieve the data and coordinate information. 
        """

#       For convenience, allows calls where expr is not a string, in which
#        case it returns back the input field or raise an exception
#       -------------------------------------------------------------------
        if isinstance(expr ,StringTypes):
            pass # OK, will proceed to retrieve it from GrADS
        elif isinstance(expr,GaField):
            return expr # just return input
        elif isinstance(expr,ndarray):
            return expr # this is handy for 'lsq'
        else:
            raise GrADSError("input <expr> has invalid type: %s"%type(expr))

        d = self.eval(expr)
        c = self.coords()
        g = GaGrid(expr,coords=c)

        Data = reshape(d,c.shape)
        F = GaField(Data,mask=(Data==c.undef),name=expr,grid=g)

        return F

#........................................................................
    def env(self, query='all'):
        """
        Query and return the GrADS dimension and display environment.
        This function is designed to make a new query every time it is
        called in order to avoid problems when assuming the last known
        state has not changed. A snapshot of the environment at a specific
        time can be saved by assigning a variable to a call of this function.
        """
        return GaEnv(self, query)

    def move_pointer(self, marker, encoding='utf-8', verbose=False):
        """
        Move the GrADS stream pointer to the given marker.
        The marker only has to match a portion of a line of output.
        Additional Args:
            encoding: Expected character encoding of the GrADS output
        """
        out = ''
        while marker not in out:
            out=self.filter_output(self.Reader.readline().decode(encoding))
            if verbose:
                print(out)
            if len(out) == 0:
                raise GrADSError("GrADS terminated.")
        return

    def py3exp(self,expr,dx=None,dy=None):
        """
        Export a GrADS field to a Numpy array. Since only up to 2-dimensional
        data can be written out by GrADS, requesting arrays of rank > 2 will be
        less efficient than defining the same array in GrADS.
        Args:
            expr: GrADS expression representing the field to be exported.
        """
        # Get the current environment
        qc=self.query('ctlinfo')

        env = self.env()
        
        env_orig=env
        dh = self.query("dims",Quiet=True)
        grid = GaGrid(expr)        
        grid.denv = dh
        grid.qc   = qc
        grid.meta = zeros((dh.nt,dh.nz,20),dtype=float32)
        ndims=Namespace(nt=qc.nt,nz=qc.nz,nx=qc.nx,ny=qc.ny)
        if len(qc.t0)==15:
            tstr='%H:%MZ%d%b%Y'
        elif len(qc.t0)==12:
            tstr='%HZ%d%b%Y'
        else: tstr=''
        try:
            t2=dt.datetime.strptime(qc.t0,tstr)
            grid.tyme = np.array([t2+dt.timedelta(minutes=qc.dt*i) for i in range(qc.nt)])
        except:
            grid.tyme = [] 
        grid.time = []
        #grid.lev = zeros(dh.nz,dtype=float32)
        if hasattr(qc,'x0'):
            #grid.lon = np.array([qc.x0+qc.dx*i for i in range(qc.nx)]) 
            if not dx: dx=qc.dx
            grid.lon=np.arange(qc.x0,qc.x0+qc.dx*qc.nx,dx)
            if not isinstance(env.lon,(tuple,list)): grid.lon=[env.lon]
            else: 
                grid.lon=np.arange(env.lon[0],env.lon[1]+dx,dx)
                if grid.lon[-1] > env.lon[1]:
                    grid.lon=grid.lon[:-1]
            #else: grid.lon=np.arange(env.lon[0],env.lon[1],dx)
            env.dx=dx; env.nx=len(grid.lon)
            ndims.nx=len(grid.lon)
        elif hasattr(qc,'xlevs'):
            grid.lon = np.array(qc.xlevs)
        if hasattr(qc,'y0'):
            #grid.lat = np.array([qc.y0+qc.dy*i for i in range(qc.ny)])
            if not dy: dy=qc.dy
            grid.lat=np.arange(qc.y0,qc.y0+qc.dy*qc.ny,dy)
            if  not isinstance(env.lat,(tuple,list)): grid.lat=[env.lat]
            else: 
                grid.lat=np.arange(env.lat[0],env.lat[1]+dy,dy)
                if grid.lat[-1] > env.lat[1]:
                    grid.lat=grid.lat[:-1]
            env.dy=dy; env.ny=len(grid.lat)
            ndims.ny=len(grid.lat)
        elif hasattr(qc,'ylevs'):
            grid.lat = np.array(qc.ylevs)
        #grid.tyme = np.array([t2+dt.timedelta(minutes=qc.dt*i) for i in range(qc.nt)])
        if hasattr(qc,'z0'):
            grid.lev = np.array([qc.z0+qc.dz*i for i in range(qc.nz)])
        elif hasattr(qc,'zlevs'):
            grid.lev = np.array(qc.zlevs)
        else:
            grid.lev = []
        dimnames = ('x','y','z','t','e') # ordered by GrADS read efficiency
        # Detect which dimensions are varying
        dims = [dim for dim in dimnames if not getattr(env, dim+'fixed')]
        # We can only display/output data from GrADS up to 2 dimensions at a
        # time, so for rank > 2, we must fix the extra dimensions. For best
        # efficiency, always select the two fastest dimensions to vary.
        varying, fixed = dims[:2], dims[2:]
        # Varying dimensions must be ordered identically to GrADS fwrite output
        fwrite_order = ['z','y','x','t','e']
        varying.sort(key=lambda dim: fwrite_order.index(dim))
        output_dims = varying + fixed
        # For common cases, it is desirable to enforce a certain dimension
        # order in the output array for the first two axes
        output_orders2D = OrderedDict([
            ('xy', ['y','x']), ('xz', ['z','x']), ('yz', ['z','y']),
            ('xt', ['t','x']), ('yt', ['y','t']), ('zt', ['z','t'])
        ])
        # Check for 2D base dimensions in order of preference
        for first2, order in output_orders2D.items():
            if set(first2).issubset(dims):
                ordered_dims = order + [d for d in dims if d not in order]
                break
        else:
            ordered_dims = dims
        # Read data into Numpy array
        if len(dims) <= 2:
            dimlengths = [getattr(env, 'n'+dim) for dim in varying]
            #arr = self._read_array(expr, varying)
            arr = self._read_array(expr, dimlengths)
        else:
            dimvals = {}
            for dim in dims:
                mn, mx = getattr(env, dim+'i')
                dimvals[dim] = range(mn, mx+1)
            # Sets of fixed coordinates for which to request arrays while the
            # first two (most efficient) dimensions vary
            coordinates = product(*[dimvals[dim] for dim in fixed])
            arr = None # Need to wait to define until we know shape of arr1D
            for coords in coordinates:
                # Set fixed dimemsions and get array indices
                idx = []
                for dim, c in zip(fixed, coords):
                    self.cmd('set {dim} {c}'.format(dim=dim, c=c))
                    idx.append(dimvals[dim].index(c))
                # Get 2D array
                dimlengths = [getattr(env, 'n'+dim) for dim in varying]
                arr2D = self._read_array(expr, dimlengths)
                # Define full data array
                if arr is None:
                    arr = np.zeros(arr2D.shape + tuple(len(dimvals[d]) for d in fixed))
                # Assign data along first two dimensions
                arr[(slice(None), slice(None)) + tuple(idx)] = arr2D
        # Re-order axes if necessary
        axes = [(i, output_dims.index(d)) for i, d in zip(range(len(dims)), ordered_dims)]
        swapped = []
        for a1, a2 in axes:
            pair = sorted([a1, a2])
            if a1 != a2 and pair not in swapped:
                arr = np.swapaxes(arr, a1, a2)
                swapped.append(pair)
        # Restore original GrADS dimension environment
        for dim in dims:
            mn, mx = getattr(env_orig, dim)
            self.cmd('set {dim} {mn} {mx}'.format(dim=dim, mn=mn, mx=mx))
        self.flush()
        return arr,env.undef,grid

    def _read_array(self, expr, dimlengths):
        """
        Read a GrADS field into a Numpy array. The rank of the array must
        be 2 or less.
        Args:
            expr: GrADS expression representing the field to be read.
            dims: List of GrADS varying dimension names defining the
                  space occupied by expr.
        """
        encoding = 'latin-1'
        env = self.env()
        # Enable GrADS binary output to stream
        self.cmd('set gxout fwrite')
        self.cmd('set fwrite -st -')
        # Don't block output here so we can intercept the data stream
        self.cmd('display '+expr, Block=False)
        # Move stream pointer to '<FWRITE>'
        self.move_pointer('<FWRITE>', encoding=encoding, verbose=False)
        # Read binary data from stream
        handle = io.BytesIO()
        chsize = 4096 # Read data in 512 byte chunks
        rcpattern = re.compile(b'\\n\<RC\>\s?-?\d+\s?\<\/RC\>') # pattern of RC tag
        fwritepattern=re.compile(b'\<FWRITE\>[\\n]*')
        chunk = self.p.stdout.read(chsize)
        fwritematch=fwritepattern.search(chunk)
        endmatch = rcpattern.search(chunk)
        
        flag=True
        if endmatch:
            chunk=chunk[:endmatch.span()[0]]
            flag=False
        if fwritematch:
            handle.write(chunk[fwritematch.span()[1]:])
        else:
            handle.write(chunk)
        while flag:
            chunk = self.p.stdout.read(chsize)

            # We know we're at the end when we encounter a return code wrapped
            # in RC tags, immediately following a newline. (\n<RC> {number} </RC>)
            # Must be very precise in detecting this because '<RC>' by itself
            # can appear in a binary data stream.
            endmatch = rcpattern.search(chunk)
            fwritematch=fwritepattern.search(chunk)
            if endmatch:
                # Cut out whatever data precedes the <RC> tag
                handle.write(chunk[:endmatch.span()[0]])
                # The ending character of the last chunk is arbitrary,
                # we only know that <RC> is in it.
                # Thus, need to flush GrADS pipes to avoid hanging
                # and reset the pointer to the next marker.
                self.flush(encoding=encoding)
                break
            elif fwritematch:
                print(chunk[:fwritematch.span()[1]])
                handle.write(chunk[fwritematch.span()[1]:])
            else:
                handle.write(chunk)
        # If GrADS is sane, normal behavior is to return the array of grid points
        # big enough to completely enclose or overlap the set domain.
        #dimlengths = [getattr(env, 'n'+dim) for dim in dims]
        
        guess_shape = tuple(dimlengths)
        guess_size = int(np.prod(guess_shape))
        try:
            # Convert binary data to 32-bit floats
            arr = np.fromstring(handle.getvalue(), dtype=np.float32)
        except:
            print(handle.getvalue()[:50] )
            raise GrADSError('Problems occurred while exporting GrADS expression: '+expr
                               +'\nCommon reasons:'
                               +'\n\t1) Dimensions which are fixed/varying in the expression '
                               +'\n\t   must be fixed/varying in the GrADS environment.'
                               +'\n\t2) One or more of your GrADS dimensions may extend out of bounds.'
                               )
        # If all is sane and expected
        if arr.size == guess_size:
            shape = guess_shape
        else:
            # For whatever reason, GrADS will sometimes return a grid offset
            # by an index or two from what the dimension environment says it
            # should be (e.g. nx*ny for an x-y field). To work around this,
            # test a few perturbations around the expected size of a single
            # dimension at a time and see if any of them work.
            possible_shapes = []
            dim_ranges = []
            for n in dimlengths:
                if n > 2:
                    r = range(n-2, n+3)
                else:
                    r = range(1, n+3)
                dim_ranges.append(r)
            # Remember, dim order determines how the shape tuples here are ordered
            possible_shapes = list(product(*dim_ranges))
            possible_sizes = [np.prod(shape) for shape in possible_shapes]
            # Actual shape of the grid. This assumes that if multiple possible
            # shapes have the same size, the first one that works is correct.
            # This will not always be true...blame GrADS for having unpredictable
            # grid sizes
            if arr.size not in possible_sizes:
                shape=next(((n,int(arr.size/n)) for n in dimlengths if arr.size%n==0),None)
            else:
                shape = possible_shapes[possible_sizes.index(arr.size)]
            
        arr = arr.reshape(shape)
        
        #arr[arr == self.MISSING] = np.nan
        # Close stream
        self.cmd('disable fwrite')
        # Restore gxout settings, assuming typical 2D scalar field plot
        self.cmd('set gxout '+env.gx2Dscalar)
        return arr


#........................................................................

    def eof ( self, expr, transf='anomaly', metric='area', keep=None):
        """ 
        Given a GrADS generalized expression *expr*, calculates Empirical 
        Orthogonal Functions (EOFS) using Singular Value Decomposition (SVD). 
        
            V, d, c = self.eof(expr)
        
        where

            V  ---  A specialized GrADS Field holding eigenvectors
                    with *add* offset and *scale* factors to aid
                    subsequent decomposition in terms of V
            d  ---  NumPy array with eigenvalues
            c  ---  NumPy array with principal components

        The other optional parameters are:

        transf     
            Type of pre-processing transform to be applied:
            None     ---  time series as is
            anomaly  ---  remove time mean
            z-score  ---  remove time mean and divide 
                          by standard deviation
        metric    
            Determines whether to scale the timeseries prior
            to calculation of the EOFs; this is equivalent to 
            choosing a particular norm. Acceptable values are:
            None   ---  do not scale
            'area' ---  multiply by cos(lat), the default

        keep
            How many eigenvectors to keep:
            None  ---  in this case keep as many vectors as 
                       there are timesteps (nt) 
            n     ---  keep "n" eigenvectors

        Notice that *expr* on input is a *generalized expression* in the
        sense that it can contain a string with a GrADS expression to be
        evaluated or a valid GrADS field. See method *exp* for additional
        information.

        IMPORTANT: the input (masked) array mask (undef values) must be
                   the same for all times.
        
        """

#       At least 2 time steps
#       ---------------------
        dh = self.query("dims",Quiet=True)
        if dh.nt < 2:
            raise GrADSError(
                  'need at least 2 time steps for EOFS but got nt=%d'%dh.nt)
        nt, nz, ny, nx = (dh.nt, dh.nz, dh.ny, dh.nx)

#       Export N-dimensional array
#       --------------------------
        u = self.exp(expr)
        g = u.grid

#       Reshape as 4D
#       -------------
        nx = len(g.lon) # may differ from dh.nx
        ny = len(g.lat) # may differ from dh.ny
        u = u.reshape(nt,nz,ny,nx)

#       Remove time mean if necessary
#       -----------------------------
        offset = ma.zeros((nz,ny,nx),dtype=float32) # place holder
        if transf==None:
            pass
        elif transf=='anomaly' or transf=='z-score':
            offset = average(u,axis=0)
            u = u - offset[newaxis,:,:,:]
        else:
            raise GrADSError('Unknown transf <%s>'%transf)
    
#       Scale by stdv if z-scores required
#       ----------------------------------
        scale = ma.ones((nz,ny,nx),dtype=float32) # place holder
        if transf=='z-score':
            scale = sqrt(average(u*u,axis=0))
            u = u / scale[newaxis,:,:,:]

#       Apply metric if so desired
#       Note: may need delp for cases with nz>1
#       ---------------------------------------
        if metric=='area':
            factor = sqrt(cos(pi*g.lat/180.))
            u = u * factor[newaxis,newaxis,:,newaxis]
            scale = scale * factor[newaxis,newaxis,:,newaxis]

#       Singular value decomposition, reuse u
#       -------------------------------------
        I = u.mask[0,:,:,:]==False  # un-masked values
        fill_value = u.fill_value   # save it for later
        pc, d, u = svd(u.data[:,I],full_matrices=0)

#       Trim eigenvectors
#       -----------------
        if keep==None:
            nv = nt
        else:
            nv = min(keep,nt)
            u = u[0:nv,:]
            d = d[0:nv]
            pc = pc[:,0:nv]

#       Adjust grid properties
#       ----------------------
        g.dims[0] = 'eof'
        g.time = arange(nv)
        g.eof = arange(nv)

#       Eigenvalues/coefficients
#       ------------------------
        d = d * d / (nt - 1)
        pc = (nt -1) * pc.transpose()

#       Normalize eigenvectors
#       ----------------------
        for i in range(nv):
            vnorm = _norm(u[i,:])
            u[i,:] = u[i,:] / vnorm
            pc[i,:] = pc[i,:] * vnorm
    
        # Scatter eigenvectors
        # --------------------
        g.meta = g.meta[0:nv] 
        u = _scatter(u,I,(nv,nz,ny,nx),fill_value)

#       Let's make sure "u" is a bonafide GaGield
#       -----------------------------------------
        u = GaField(u.data, name=expr, grid=g, mask=u.mask)
        u.offset = offset.squeeze()
        u.scale = scale.squeeze()

#       Note: since GrADS v1 does not know about e-dimensions yet,
#       we let it think that the EOF dimension is the time dimension

#       All done
#       --------
        return (u, d, pc)

#.....................................................................

    def lsq (self, y_expr, x_exprs, Bias=False, Mask=None):
        """
        Given a target GrADS expression *y_expr* and a tuple of predictor
        GrADS expressions *x_exprs*, returns a NumPy array with linear 
        regression coefficients 

            c, info = self.lsq(y_expr, x_exprs)

        where *info* contains information about the minimization:

            info.residuals  ---  sum square of residuals
            info.rank       ---  rank of the predictor matrix
            info.s          ---  singular values of predictor matrix

        When Bias=False (default) the residual

            y - c[0] * x[:,0] + c[1] * x[:,1] + ... + c[n] * x[:,N-1]

        is minimized in the last square sense, where *x* and *y* are
        NumPy arrays associated with the following GrADS Fields:

            Y = self.exp(y_expr)
            X[:,n] = self.exp(x_exprs[n]),  n = 0, ..., N-1

       When Bias=True, the additional predictor array 

            x[N] = ones(y)

       is included in the calculation, resulting in an output array of
       size N+1. This is equivalent to allowing for an *intercept* in
       the regression equation.

       The optional Mask is a 1D logical array with the location of the
       data to be included (see compress() method).

       On input, all expressions are *generalized expressions* in the
       sense that they can contain a string with a GrADS expression to be
       evaluated or a valid GrADS field. See method *expr* for additional
       information.

       """

        N = len(x_exprs)
        if N<1:
            raise GrADSError(
                'expecting at least one predictor but got %d'%N)
        if Bias: N = N + 1
        
#       Retrieve target
#       ---------------
        f = self.exp(y_expr)
        y = f.ravel()
        if Mask!=None:  
            y = y.compress(Mask)
        M = y.size

#       Retrieve predictors
#       -------------------
        X = ones((M,N),dtype=float32)
        for n in range(len(x_exprs)):
            f = self.exp(x_exprs[n])
            x = f.ravel()
            if Mask!=None: 
                x = x.compress(Mask)
            X[:,n] = x

#       Perform LS minimization
#       -----------------------
        info = GaHandle('lsq')
        (c, info.residuals, info.rank, info.s) = lstsq(X,y)

#       All done
#       --------
        return (c, info)

#   ..................................................................

    def _interpXY ( self, expr, lons, lats, levs=None, dh=None, **kwopts):
        """
        Evaluates GrADS expression (or GaField) and interpolates it to
        the the (longitude,latitude) points given the input arrays
        (lons,lats) on input. Both x/y dimensions must be
        varying. When the z-dimenson is varying as well a curtain
        slice is returned. For now, the time dimension must be fixed.
        Example:

          tau, levs = ga.interp('duexttau',lons,lats)

        where *levs* is an 1D array with the versical levels. The optional
        **kwopts arguments are passwd to the interpolate() function.

        Note: the basemap interpolation routine requires longitudes in
        the range [-180,180]. When *expr* is a string the longitudes are
        set to the correct range. However, when *expr* is a GaField
        the user must make sure this is the case or undefs may result.
        """

#       Check dim environment
#       ---------------------
        if dh==None:
            dh = self.query("dims", Quiet=True)
        if dh.nx==1 or dh.ny==1:
            raise GrADSError(
            "expecting varying x/y dimensions but got (nx,ny) = (%d,%d)"\
            %(dh.nx,dh.ny))
        if dh.nt>1:
            raise GrADSError(
            "sorry, cannot interpolate with varying time dimension")

#       Evaluate GrADS expression
#       -------------------------
        if dh.lon[0]>180. or dh.lon[1]>180:
            self.cmd('set lon -180 180',Quiet=True) # assume global grid
        Z = self.exp(expr)
        g = Z.grid
        self.cmd('set x %s %s'%dh.x,Quiet=True)

#       Loop over vertical levels
#       -------------------------
        n = size(lons)
        lon_, lat_ = reshape(lons,(n,1)), reshape(lats,(n,1))
        if len(Z.shape)==2:
            y = interpolate(Z, g.lon, g.lat, lon_, lat_, 
                       masked=True, **kwopts)
        else: 
            y = ma.masked_array(zeros((n,1,dh.nz),dtype=float32))
            for z in range(dh.nz): # because Interp needs 2D arrays on input
                y[:,:,z] = interpolate(Z[z], g.lon, g.lat, lon_, lat_,
                                       masked=True, **kwopts)
                
#       Return array with same shape as the input lons/lats, 
#        with possibly an additional dimension in case the
#        z-dimension is varying
#       ----------------------------------------------------
        y = ma.masked_array(y,dtype=float32) # for consistency of types
        return (y.squeeze(),g.lev)

#   ..................................................................

    def sampleXY ( self, expr, lons, lats, levs=None, dh=None, **kwopts):
        """
        Evaluates GrADS expression (or GaField) and interpolates it to
        the the (longitude,latitude) points given the input arrays
        (lons,lats) on input. Both x/y dimensions must be
        varying. When the z-dimenson is varying as well a curtain
        slice is returned. The output is a special case of a GaField,
        where the first axis contains the "observational dimension". 
        
        Example:

          tau = ga.sampleXY('duexttau',lons,lats)

        The trajectory coordinates are (lons,lats) returned in the
        grid atrributes, e.g., tau.grid.lon, tau.grid.lat.

        The optional **kwopts arguments are passed to the
        interpolate() function.

        Note: the basemap interpolation routine requires longitudes in
        the range [-180,180]. When *expr* is a string the longitudes are
        set to the correct range. However, when *expr* is a GaField
        the user must make sure this is the case or undefs may result.
        """

        # Inputs must be 1D arrays
        # ------------------------
        if len(lons.shape)!=1 or len(lats.shape)!=1:
            raise GrADSError("lons, lats, time must be 1D arrays")
        
        
        # Retrieve dimension environment
        # ------------------------------
        dh = self.query("dims", Quiet=True) 

        # Loop over time, performing interpolation
        # ----------------------------------------
        g = GaGrid("sampleXY")
        g.time = []
        V  = ma.masked_array(zeros((len(lons),dh.nt,dh.nz)),dtype=float32)
        for t in dh.ti:
            n = t - dh.ti[0]
            self.cmd('set t %d'%t, Quiet=True)
            v, g.lev = self._interpXY ( expr, lons, lats, 
                                           levs=levs, 
                                           **kwopts)
            if len(v.shape)==1:
                V[:,n,0] = v
            else:
                V[:,n,:] = v
            qh =self.query("time",Quiet=True)
            g.time.append(qh.t1)

        g.dims = ['obs',]
        if dh.nt>1:
            g.dims.append('time')
        if dh.nz>1:
            g.dims.append('lev')
        
        g.lon, g.lat = (lons, lats) # "obs" coordinates
        g.tyme = array([gat2dt(t) for t in g.time])

#       Restore dimension environment
#       -----------------------------
        self.setdim(dh)
        V = V.squeeze()
        return GaField(V.data, name=expr, grid=g, 
                       mask=V.mask, dtype=float32)
#   ..................................................................

    def sampleXYT ( self, expr, lons, lats, tyme=None,
                    levs=None, dh=None, Verbose=False, **kwopts):
        """
        Evaluates GrADS expression (or GaField) and interpolates it to
        the the (longitude,latitude,time) points given the input arrays
        (lons,lats,tyme) on input. Both x/y dimensions must be
        varying. When the z-dimenson is varying as well a curtain
        slice is returned. If *tyme* is not specified it reverts to
        method sampleXY(). The output is a special case of a GaField,
        where the first axis contains the "observational dimension". 
        
        Example:

          tau = ga.sampleXYT('duexttau',lons,lats,

        The trajectory coordinates (lons,lats,tyme) are returned in
        the grid atrributes, e.g., tau.grid.lon, tau.grid.lat,
        tau.grid.tyme.
    
        The optional **kwopts arguments are passed to the
        interpolate() function.  Notice that *tyme* is an array of
        datetime objects.

        Note: the basemap interpolation routine requires longitudes in
        the range [-180,180]. When *expr* is a string the longitudes are
        set to the correct range. However, when *expr* is a GaField
        the user must make sure this is the case or undefs may result.
        """

        # Revert back to InterpXY if no time is specified
        # -----------------------------------------------
        if tyme is None:
            return self.sampleXY ( expr, lons, lats, 
                                   levs=levs, dh=dh, 
                                   **kwopts)

        # Inputs must be 1D arrays
        # ------------------------
        if len(lons.shape)!=1 or len(lats.shape)!=1 or len(tyme.shape)!=1:
            raise GrADSError("lons, lats, tyme must be 1D arrays")
        
        # Retrieve dimension environment
        # ------------------------------
        dh = self.query("dims", Quiet=True) 

        # Find GrADS times bracketing the input time array
        # ------------------------------------------------
        self.cmd('set time %s'%dt2gat(tyme[0]),Quiet=True)
        qh = self.query("dims",Quiet=True)
        tbeg = int(qh.t[0])
        if tyme[0] < gat2dt(qh.time[0]):
            tbeg = tbeg - 1
        self.cmd('set time %s'%dt2gat(tyme[-1]),Quiet=True)
        qh = self.query("dims",Quiet=True)
        tend = int(qh.t[0])
        if tyme[-1] > gat2dt(qh.time[0]):
            tend = tend + 1

        # Check if (tbeg,tend) is in range of default file
        # ------------------------------------------------
        fh = self.query("file",Quiet=True)
        if tbeg<1 or tbeg>fh.nt:
            raise GrADSError("(tbeg,tend) outside of range (1,%d)"%fh.nt)

        # Find time step
        # --------------
        dt = self._getDatetime(tbeg+1) - self._getDatetime(tbeg)
        dt_secs = dt.total_seconds()
        
        # Loop over time, producing XY interpolation at each time
        # -------------------------------------------------------
        V, I = [], []
        for t in range(tbeg,tend+1):
            now = self._getDatetime(t) # grads time is set to t
            if Verbose: print(" [] XY Interpolating at ", now)
            i = (tyme>=now-dt) & (tyme<=now+dt)
            if any(i):
                self._tightDomain(lons[i],lats[i]) # minimize I/O
                v, levs = self._interpXY(expr, lons[i], lats[i], levs=levs, **kwopts)
            else:
                v = None
            V.append(v)
            I.append(i)
            
        # Now perform the time interpolation
        # ----------------------------------
        N = len(lons) 
        if len(levs)>1:
            shp = [N,len(levs)]
        else:
            shp = [N,]
        v  = ma.masked_array(zeros(shp),dtype=float32)
        v1, v2 = v.copy(), v.copy() # scratch space
        n = 0
        for t in range(tbeg,tend):
            now = self._getDatetime(t) 
            v1[I[n]], v2[I[n+1]] = V[n], V[n+1]
            j = (tyme>=now) & (tyme<=now+dt)
            if any(j): 
                a = array([r.total_seconds()/dt_secs for r in tyme[j]-now],dtype=float32) 
                if len(shp)==2: # has vertical levels
                    a = tile(a,(shp[1],1)).T # replicate array
                v[j] = (1-a) * v1[j] + a * v2[j]
            n += 1

        # Grid
        # ----
        g = GaGrid("sampleXYT")
        g.lev = levs
        g.lon, g.lat, g.tyme = (lons, lats, tyme)
        g.time = array([dt2gat(t) for t in g.tyme])
        g.dims = ['obs',]
        if dh.nz>1:
            g.dims.append('lev')
        
        # Restore dimension environment
        # -----------------------------
        self.setdim(dh)
        
        return GaField(v.data, name=expr, grid=g, 
                       mask=v.mask, dtype=float32)
    
#   ..................................................................

    def sampleKML ( self, expr, kml_filename,
                    speed=90.,t0=None,metric=True,noSinglePoint=True,
                    **kwopts):
        """
        Evaluates GrADS expression (or GaField) and interpolates it to
        the the (longitude,latitude) points given in the input KML
        file (from Google Maps).

        On input, *speed* is the average speed, in km/h if *metric* is
        True, or miles/h otherwise. The initial time *t0* (a datetime
        object in UTC, not local time) defaults to now if not specified.
        
        Both x/y dimensions must be varying. When the z-dimenson is
        varying as well a curtain slice is returned.
        
        The output is a special case of a GaField, where the first
        axis contains the"observational dimension".

        Example:

          tau = ga.sampleKML('duexttau','directions_to_acadia.kml')

        The route coordinates are returned in the grid atrributes,
        e.g., tau.grid.lon, tau.grid.lat, tau.grid.tyme, tau.grid.dst,
        where *dst* is the distance from the first point in the route.
        
         The optional **kwopts arguments are passed to the sampleXYT()
        method.

        """

        kml = SimpleKML(kml_filename)
        lon, lat, dst, tyme = kml.getCoords(speed=speed,t0=t0,metric=metric,
                                            noSinglePoint=noSinglePoint)
        var = self.sampleXYT(expr,lon,lat,tyme,**kwopts)
        var.grid.dst = dst # distance from start
        
        return var
    
#   ..................................................................
    interp = _interpXY  # deprecated, use sample functions instead
#   ..................................................................

    def _getDatetime(self,t):
        """
        Return datetime given grads time index "t" or "time"
        Side effect: the grads time is set to "t".
        """
        if type(t) == type(1):
            self.cmd('set t %d'%t,Quiet=True)
        elif type(t) == type("abc"):
            self.cmd('set time %s'%t,Quiet=True)
        qh = self.query("dims",Quiet=True)
        return gat2dt(qh.time[0])

#   ..................................................................

    def _tightDomain(self,lons,lats):
        """
        Reduce the (lat,lon_ domain as to bracket the coordinates
        on input.
        Side effect: dimension environment is modified.
        """
        self.cmd('set lon %f %f'%(lons.min(),lons.max()),Quiet=True)
        self.cmd('set lat %f %f'%(lats.min(),lats.max()),Quiet=True)
        fh = self.query("file",Quiet=True)
        qh = self.query("dims",Quiet=True)
        x1, x2 = (qh.xi[0]-1,qh.xi[1]+1)
        y1, y2 = (max(1,qh.yi[0]-1),min(fh.ny,qh.yi[1]+1)) # in [1,ny]
        self.cmd('set x %d %d'%(x1,x2),Quiet=True)
        self.cmd('set y %d %d'%(y1,y2),Quiet=True)
   
#.....................................................................

def _norm(x):
    """
    L-2 norm, internal use
    """
    return sqrt(inner(x,x))

def _scatter(u,I,shp,fill_value):
    """
    Scatter input array according to index mask I.
    """
    v = ma.masked_array(data=zeros(shp,dtype='float32')+fill_value,
                        mask=ones(shp,dtype='bool'),
                        fill_value=fill_value)
    v = v.squeeze()
    v.data[:,I.squeeze()] = u[:,:]
    v.mask[:,I.squeeze()] = False
    
    return v

def interpolate(datain,xin,yin,xout,yout,checkbounds=False,masked=False,order=1):
    """
    Note: This function borrowed from basemap. Reproduced here to remove basemap
          dependency.
    
    Interpolate data (``datain``) on a rectilinear grid (with x = ``xin``
    y = ``yin``) to a grid with x = ``xout``, y= ``yout``.

    .. tabularcolumns:: |l|L|

    ==============   ====================================================
    Arguments        Description
    ==============   ====================================================
    datain           a rank-2 array with 1st dimension corresponding to
                     y, 2nd dimension x.
    xin, yin         rank-1 arrays containing x and y of
                     datain grid in increasing order.
    xout, yout       rank-2 arrays containing x and y of desired output grid.
    ==============   ====================================================

    .. tabularcolumns:: |l|L|

    ==============   ====================================================
    Keywords         Description
    ==============   ====================================================
    checkbounds      If True, values of xout and yout are checked to see
                     that they lie within the range specified by xin
                     and xin.
                     If False, and xout,yout are outside xin,yin,
                     interpolated values will be clipped to values on
                     boundary of input grid (xin,yin)
                     Default is False.
    masked           If True, points outside the range of xin and yin
                     are masked (in a masked array).
                     If masked is set to a number, then
                     points outside the range of xin and yin will be
                     set to that number. Default False.
    order            0 for nearest-neighbor interpolation, 1 for
                     bilinear interpolation, 3 for cublic spline
                     (default 1). order=3 requires scipy.ndimage.
    ==============   ====================================================

    .. note::
     If datain is a masked array and order=1 (bilinear interpolation) is
     used, elements of dataout will be masked if any of the four surrounding
     points in datain are masked.  To avoid this, do the interpolation in two
     passes, first with order=1 (producing dataout1), then with order=0
     (producing dataout2).  Then replace all the masked values in dataout1
     with the corresponding elements in dataout2 (using numpy.where).
     This effectively uses nearest neighbor interpolation if any of the
     four surrounding points in datain are masked, and bilinear interpolation
     otherwise.

    Returns ``dataout``, the interpolated data on the grid ``xout, yout``.
    """
    import numpy as np
    import numpy.ma as ma
    # xin and yin must be monotonically increasing.
    if xin[-1]-xin[0] < 0 or yin[-1]-yin[0] < 0:
        raise ValueError('xin and yin must be increasing!')
    if xout.shape != yout.shape:
        raise ValueError('xout and yout must have same shape!')
    # check that xout,yout are
    # within region defined by xin,yin.
    if checkbounds:
        if xout.min() < xin.min() or \
           xout.max() > xin.max() or \
           yout.min() < yin.min() or \
           yout.max() > yin.max():
            raise ValueError('yout or xout outside range of yin or xin')
    # compute grid coordinates of output grid.
    delx = xin[1:]-xin[0:-1]
    dely = yin[1:]-yin[0:-1]
    if max(delx)-min(delx) < 1.e-4 and max(dely)-min(dely) < 1.e-4:
        # regular input grid.
        xcoords = (len(xin)-1)*(xout-xin[0])/(xin[-1]-xin[0])
        ycoords = (len(yin)-1)*(yout-yin[0])/(yin[-1]-yin[0])
    else:
        # irregular (but still rectilinear) input grid.
        xoutflat = xout.flatten(); youtflat = yout.flatten()
        ix = (np.searchsorted(xin,xoutflat)-1).tolist()
        iy = (np.searchsorted(yin,youtflat)-1).tolist()
        xoutflat = xoutflat.tolist(); xin = xin.tolist()
        youtflat = youtflat.tolist(); yin = yin.tolist()
        xcoords = []; ycoords = []
        for n,i in enumerate(ix):
            if i < 0:
                xcoords.append(-1) # outside of range on xin (lower end)
            elif i >= len(xin)-1:
                xcoords.append(len(xin)) # outside range on upper end.
            else:
                xcoords.append(float(i)+(xoutflat[n]-xin[i])/(xin[i+1]-xin[i]))
        for m,j in enumerate(iy):
            if j < 0:
                ycoords.append(-1) # outside of range of yin (on lower end)
            elif j >= len(yin)-1:
                ycoords.append(len(yin)) # outside range on upper end
            else:
                ycoords.append(float(j)+(youtflat[m]-yin[j])/(yin[j+1]-yin[j]))
        xcoords = np.reshape(xcoords,xout.shape)
        ycoords = np.reshape(ycoords,yout.shape)
    # data outside range xin,yin will be clipped to
    # values on boundary.
    if masked:
        xmask = np.logical_or(np.less(xcoords,0),np.greater(xcoords,len(xin)-1))
        ymask = np.logical_or(np.less(ycoords,0),np.greater(ycoords,len(yin)-1))
        xymask = np.logical_or(xmask,ymask)
    xcoords = np.clip(xcoords,0,len(xin)-1)
    ycoords = np.clip(ycoords,0,len(yin)-1)
    # interpolate to output grid using bilinear interpolation.
    if order == 1:
        xi = xcoords.astype(np.int32)
        yi = ycoords.astype(np.int32)
        xip1 = xi+1
        yip1 = yi+1
        xip1 = np.clip(xip1,0,len(xin)-1)
        yip1 = np.clip(yip1,0,len(yin)-1)
        delx = xcoords-xi.astype(np.float32)
        dely = ycoords-yi.astype(np.float32)
        dataout = (1.-delx)*(1.-dely)*datain[yi,xi] + \
                  delx*dely*datain[yip1,xip1] + \
                  (1.-delx)*dely*datain[yip1,xi] + \
                  delx*(1.-dely)*datain[yi,xip1]
    elif order == 0:
        xcoordsi = np.around(xcoords).astype(np.int32)
        ycoordsi = np.around(ycoords).astype(np.int32)
        dataout = datain[ycoordsi,xcoordsi]
    elif order == 3:
        try:
            from scipy.ndimage import map_coordinates
        except ImportError:
            raise ValueError('scipy.ndimage must be installed if order=3')
        coords = [ycoords,xcoords]
        dataout = map_coordinates(datain,coords,order=3,mode='nearest')
    else:
        raise ValueError('order keyword must be 0, 1 or 3')
    if masked and isinstance(masked,bool):
        dataout = ma.masked_array(dataout)
        newmask = ma.mask_or(ma.getmask(dataout), xymask)
        dataout = ma.masked_array(dataout,mask=newmask)
    elif masked and is_scalar(masked):
        dataout = np.where(xymask,masked,dataout)
    return dataout

###############################################
#           GrADS Environment Handle          #
###############################################
class GaEnv:
    def __init__(self, ga, query='all'):
        """
        Container for holding GrADS dimension and display environment data.
        The information is derived from GrADS query commands ['dims','gxout'].
        A specific query may be requested if only one is needed. Default
        is to load all supported queries.
        """
        # Query dims
        if query in ('dims', 'all'):
            qdims, rc = ga.cmd('query dims',sendOutput=True)
            if rc > 0:
                raise GrADSError('Error running "query dims"')
            qdims=qdims[1:]
            # Current open file ID
            self.fid = int(qdims[0].split()[-1])
            # Which dimensions are varying or fixed?
            self.xfixed = 'fixed' in qdims[1]
            self.yfixed = 'fixed' in qdims[2]
            self.zfixed = 'fixed' in qdims[3]
            self.tfixed = 'fixed' in qdims[4]
            self.efixed = 'fixed' in qdims[5]

            # Get the dimension values. These are single numbers if the dimension
            # is fixed, or a tuple of (dim1, dim2) if the dimension is varying.
            # Grid coordinates x,y,z,t,e can be non-integers for varying dimensions,
            # but it is useful to have the "proper" integer grid coordinates xi,yi,zi,ti,ei.
            # If a dimension is fixed, GrADS automatically rounds non-integer dimensions
            # to the nearest integer.
            xinfo = qdims[1].split()
            if self.xfixed:
                self.lon = float(xinfo[5])
                self.x = float(xinfo[8])
                self.xi = int(np.round(self.x))
                self.nx = 1
            else:
                self.lon = (float(xinfo[5]), float(xinfo[7]))
                self.x = (float(xinfo[10]), float(xinfo[12]))
                self.xi = (int(np.floor(self.x[0])), int(np.ceil(self.x[1])))
                self.nx = self.xi[1] - self.xi[0] + 1
            
            yinfo = qdims[2].split()
            if self.yfixed:
                self.lat = float(yinfo[5])
                self.y = float(yinfo[8])
                self.yi = int(np.round(self.y))
                self.ny = 1
            else:
                self.lat = (float(yinfo[5]), float(yinfo[7]))
                self.y = (float(yinfo[10]), float(yinfo[12]))
                self.yi = (int(np.floor(self.y[0])), int(np.ceil(self.y[1])))
                self.ny = self.yi[1] - self.yi[0] + 1
            
            zinfo = qdims[3].split()
            if self.zfixed:
                self.lev = float(zinfo[5])
                self.z = float(zinfo[8])
                self.p = float(zinfo[5])
                self.zi = int(np.round(self.z))
            else:
                self.lev = (float(zinfo[5]), float(zinfo[7]))
                self.z = (float(zinfo[10]), float(zinfo[12]))
                self.p = (float(zinfo[5]), float(zinfo[7]))
                self.zi = (int(np.floor(self.z[0])), int(np.ceil(self.z[1])))
            tinfo = qdims[4].split()
            if len(tinfo[5]) > 12:
                timefmt = '%H:%MZ%d%b%Y'
            else:
                timefmt = '%HZ%d%b%Y'
            if self.tfixed:
                self.time = datetime.strptime(tinfo[5], timefmt)
                self.t = float(tinfo[8])
                self.ti = int(np.round(self.t))
            else:
                self.time = (datetime.strptime(tinfo[5], timefmt),
                             datetime.strptime(tinfo[7], timefmt))
                self.t = (float(tinfo[10]), float(tinfo[12]))
                self.ti = (int(np.floor(self.t[0])), int(np.ceil(self.t[1])))
            einfo = qdims[5].split()
            if self.efixed:
                self.e = float(einfo[8])
                self.ei = int(np.round(self.e))
            else:
                self.e = (float(einfo[10]), float(einfo[12]))
                self.ei = (int(np.floor(self.e[0])), int(np.ceil(self.e[1])))

            # Dimension lengths in the current environment.
            # Different from total dimension length in the file (see ctlinfo)
            if self.xfixed:
                self.nx = 1
            else:
                self.nx = self.xi[1] - self.xi[0] + 1
            if self.yfixed:
                self.ny = 1
            else:
                self.ny = self.yi[1] - self.yi[0] + 1
            if self.zfixed:
                self.nz = 1
            else:
                self.nz = self.zi[1] - self.zi[0] + 1
            if self.tfixed:
                self.nt = 1
            else:
                self.nt = self.ti[1] - self.ti[0] + 1
            if self.efixed:
                self.ne = 1
            else:
                self.ne = self.ei[1] - self.ei[0] + 1

            # Rank of the environment space (number of dimensions)
            self.rank = sum([not d for d in
                             [self.xfixed,self.yfixed,self.zfixed,self.tfixed,self.efixed]])

        # Query ctlinfo
        if query in ('ctlinfo', 'all'):
            qctl, rc = ga.cmd('query ctlinfo',sendOutput=True)
            qctl=qctl[1:]
            if rc > 0:
                raise GrADSError('Error running "query ctlinfo"')
            # Total dimension lengths in the file
            self.Ne = 1
            for line in qctl:
                if 'xdef ' in line or 'XDEF ' in line:
                    self.Nx = int(line.split()[1])
                elif 'ydef ' in line or 'YDEF ' in line:
                    self.Ny = int(line.split()[1])
                elif 'zdef ' in line or 'ZDEF ' in line:
                    self.Nz = int(line.split()[1])
                elif 'tdef ' in line or 'TDEF ' in line:
                    self.Nt = int(line.split()[1])
                # EDEF section may or may not be present
                elif 'edef ' in line or 'EDEF ' in line:
                    self.Ne = int(line.split()[1])
                elif 'undef' in line.lower():
                    self.undef = float(line.split()[1])
        # Query gxout
        if query in ('gxout', 'all'):
            qgxout, rc = ga.cmd('query gxout',sendOutput=True)
            qgxout=qgxout[1:]
            if rc > 0:
                raise GrADSError('Error running "query gxout"')
            # gxout defines graphics types for 1D scalar plots, 1D vector plots,
            # 2D scalar plots, and 2D vector plots.
            # Map GrADS graphic identifiers to gxout commands. Note that "gxout stat"
            # and "gxout print" do not change the output of "query gxout"
            graphicTypes = {'Contour': 'contour', 'Line': 'line', 'Barb': 'barb',
                            '16': 'shaded', '17': 'shade2b', 'Shaded': 'shade1',
                            'Vector': 'vector', 'Shapefile': 'shp', 'Bar': 'bar',
                            'Grid': 'grid', 'Grfill': 'grfill', 'Stream': 'stream',
                            'Errbar': 'errbar', 'GeoTIFF': 'geotiff', 'Fgrid': 'fgrid',
                            'ImageMap': 'imap', 'KML': 'kml', 'Linefill': 'linefill',
                             'Scatter': 'scatter', 'Fwrite': 'fwrite', '0': None}

            # Get current graphics settings
            self.gx1Dscalar = graphicTypes[qgxout[1].split()[-1]]
            self.gx1Dvector = graphicTypes[qgxout[2].split()[-1]]
            self.gx2Dscalar = graphicTypes[qgxout[3].split()[-1]]
            self.gx2Dvector = graphicTypes[qgxout[4].split()[-1]]
            stationData = qgxout[5].split()[-1]
            if stationData == '6':
                self.stationData = None
            else:
                self.stationData = stationData

        # Query gxinfo
        if query in ('gxinfo', 'all'):
            qgxinfo, rc = ga.cmd('query gxinfo',sendOutput=True)
            qgxinfo=qgxinfo[1:]
            if rc > 0:
                raise GrADSError('Error running "query gxinfo"')
            # Get page limits and the current plot's limits in page coordinates
            line1 = qgxinfo[1].split()
            self.pagewidth, self.pageheight = line1[3], line1[5]
            line2 = qgxinfo[2].split()
            self.Xplot = (float(line2[3]), float(line2[5]))
            line3 = qgxinfo[3].split()
            self.Yplot = (float(line3[3]), float(line3[5]))
        
        

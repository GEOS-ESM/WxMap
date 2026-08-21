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
This module defines GrADS specific types by subclassing numpy.ma.MaskedArray.
"""

__version__ = '1.1.0'

from copy      import deepcopy
from numpy     import ma, array
try:
    from mygrads.gahandle  import GaHandle
except Exception:
    from gahandle  import GaHandle

from datetime  import datetime
import numpy as np 

class GaGrid:
    """
    A simple class for holding GrADS coordinate variables as well as
    dimension environment information and other necessary metadata for
    exchanging information with GrADS. A GaGrid object is usually
    attached to a GaField object.
    """
    def __init__ (self, name, coords=None):
        """
        Creates an empty GaGrid object, or builds it from the GaHandle
        object *coords* usually obtained from the dimension environment
        with the GaCore method coords(), e.g.,

            coords = ga.coords()
        
        """
        self.name = name
        if coords==None:
            self.meta = None
            self.denv = None
            self.qc   = None
            self.dims = None
            self.time = None
            self.tyme = None
            self.lev  = None
            self.lat  = None
            self.lon  = None
        elif isinstance(coords,GaHandle):
            self.meta = None
            self.denv = coords.denv
            self.dims = coords.dims
            self.time = coords.time
            self.tyme = array([_gat2dt(t) for t in coords.time])
            self.lev  = array(coords.lev)
            self.lat  = array(coords.lat)
            self.lon  = array(coords.lon)
        else:
            raise TypeError( "coords must be a GaHandle object")

    def copy(self):
        """
        Returns a deep copy of GaGrid object.
        """
        g = GaGrid(self.name)
        for a in self.__dict__:
            v = self.__dict__[a]
            if v is not None:
                g.__dict__[a] = deepcopy(v)
        return g

class GaField(ma.MaskedArray):
    """
    This is GraDS version of a n-dimensional array: a masked array with
    a *grid* containing coordinate/dimension information attached to it.

    As a MaskedArray, GaField objects may possibly have masked values.
    Masked values of 1 exclude the corresponding element from any
    computation.
  
    Construction:
        x = GaField (data, name=None, Grid=None, 
                     dtype=None, copy=True, order=False,
                     mask = nomask, fill_value=None)
  
    If copy=False, every effort is made not to copy the data:
        If data is a MaskedArray, and argument mask=nomask,
        then the candidate data is data.data and the
        mask used is data.mask. If data is a numeric array,
        it is used as the candidate raw data.
        If dtype is not None and
        is != data.dtype.char then a data copy is required.
        Otherwise, the candidate is used.
  
    If a data copy is required, raw data stored is the result of:
    numeric.array(data, dtype=dtype.char, copy=copy)
  
    If mask is nomask there are no masked values. Otherwise mask must
    be convertible to an array of booleans with the same shape as x.
  
    fill_value is used to fill in masked values when necessary,
    such as when printing and in method/function filled().
    The fill_value is not used for computation within this module.

    """
    def __new__(
        cls,
        data,
        name=None,
        grid=None,
        mask=ma.nomask,
        **kwargs,
    ):
        obj = ma.array(
            data,
            mask=mask,
            **kwargs,
        ).view(cls)

        obj.name = name
        obj.grid = grid.copy() if grid is not None else GaGrid(name)

        return obj

    def __array_finalize__(self, obj):
        # Must run first: this is what sets up MaskedArray's own internal
        # state (_mask, _fill_value, _hardmask, ...). Skipping it leaves
        # those unset, e.g. crashing on print() with an AttributeError on
        # '_mask'.
        super().__array_finalize__(obj)
        if obj is None:
            return

        self.name = getattr(obj, "name", None)

        grid = getattr(obj, "grid", None)
        self.grid = grid.copy() if grid is not None else None

    def copy(self):
        obj = super().copy()

        # __array_finalize__ receives an intermediate plain ndarray (not
        # self) as `obj` during MaskedArray.copy(), so name/grid don't
        # propagate through it here - reassign explicitly from self instead.
        obj.name = self.name
        obj.grid = self.grid.copy() if self.grid is not None else None

        return obj

#   Use closure to overload most operations. MaskedArray's own operator
#   implementations don't consistently invoke __array_wrap__/
#   __array_finalize__ (e.g. __add__/__gt__ skip it, __abs__ doesn't), so
#   name/grid have to be reattached explicitly per operation rather than
#   relying on the general numpy subclassing hooks above.
#   ---------------------------------------
    def ga_ops ( op ):
        def wrapper(self,*args,**kwargs):
            return GaField(op(self,*args,**kwargs),
                           name=self.name,grid=self.grid)
        return wrapper

    __int__ = ga_ops(ma.MaskedArray.__int__)
    __ror__ = ga_ops(ma.MaskedArray.__ror__)
    __rsub__ = ga_ops(ma.MaskedArray.__rsub__)
    __rmul__ = ga_ops(ma.MaskedArray.__rmul__)
    __rmod__ = ga_ops(ma.MaskedArray.__rmod__)
    __rshift__ = ga_ops(ma.MaskedArray.__rshift__)

    __abs__ = ga_ops(ma.MaskedArray.__abs__)

    # __div__/__idiv__ (Python 2 division protocol) were dropped: '/' has
    # used __truediv__ since Python 3, and __div__/__idiv__ don't exist on
    # MaskedArray under newer numpy, so keeping them crashes at class
    # definition time.
    __truediv__ = ga_ops(ma.MaskedArray.__truediv__)
    __rtruediv__ = ga_ops(ma.MaskedArray.__rtruediv__)
    __itruediv__ = ga_ops(ma.MaskedArray.__itruediv__)

    __floordiv__ = ga_ops(ma.MaskedArray.__floordiv__)
    __rfloordiv__ = ga_ops(ma.MaskedArray.__rfloordiv__)
    __ifloordiv__ = ga_ops(ma.MaskedArray.__ifloordiv__)

    __isub__ = ga_ops(ma.MaskedArray.__isub__)
    __pow__ = ga_ops(ma.MaskedArray.__pow__)
    __lshift__ = ga_ops(ma.MaskedArray.__lshift__)
    __gt__ = ga_ops(ma.MaskedArray.__gt__)
    __radd__ = ga_ops(ma.MaskedArray.__radd__)
    __eq__ = ga_ops(ma.MaskedArray.__eq__)
    __rxor__ = ga_ops(ma.MaskedArray.__rxor__)
    __array__ = ga_ops(ma.MaskedArray.__array__)
    __mod__ = ga_ops(ma.MaskedArray.__mod__)
    __neg__ = ga_ops(ma.MaskedArray.__neg__)
    __iadd__ = ga_ops(ma.MaskedArray.__iadd__)
    __le__ = ga_ops(ma.MaskedArray.__le__)
    __sub__ = ga_ops(ma.MaskedArray.__sub__)
    __ge__ = ga_ops(ma.MaskedArray.__ge__)
    __and__ = ga_ops(ma.MaskedArray.__and__)
    __lt__ = ga_ops(ma.MaskedArray.__lt__)
    __rand__ = ga_ops(ma.MaskedArray.__rand__)
    __float__ = ga_ops(ma.MaskedArray.__float__)
    __ne__ = ga_ops(ma.MaskedArray.__ne__)
    __add__ = ga_ops(ma.MaskedArray.__add__)
    __imul__ = ga_ops(ma.MaskedArray.__imul__)
    __xor__ = ga_ops(ma.MaskedArray.__xor__)
    __mul__ = ga_ops(ma.MaskedArray.__mul__)
    __or__ = ga_ops(ma.MaskedArray.__or__)

    ravel = ga_ops(ma.MaskedArray.ravel)

__Months__ = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

def _gat2dt(gat):
    """
    Convert grads time to datetime.
    """
    time, date = gat.upper().split('Z')
    if time.count(':') > 0:
        h, m = time.split(":")
    else:
        h = time
        m = '0'
    mmm = date[-7:-4]
    dd, yy = date.split(mmm)
    mm = __Months__.index(mmm) + 1
    dt = datetime(int(yy),int(mm),int(dd),int(h),int(m))
    return dt

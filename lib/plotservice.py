import copy
import json
import calendar
import mydatetime as dt
from plot import *
from math import exp, log

novalue = object()

class PlotService(object):

    def __init__(self, config=None, dataservice=None, name=None):

        self.ds      = dataservice
        self.config  = config
        self.platform= 'grads'
        self.name    = self.__module__

        if name: self.name = name

#------------------------------------------------------------------------------

    def get_capabilities(self, request=None): pass

#------------------------------------------------------------------------------

    def set_name(self, name): self.name = name

#------------------------------------------------------------------------------

    def register(self, config=None, dataservice=None):

        if config is not None:
            self.config = config

        if dataservice is not None:
            self.ds = dataservice

#------------------------------------------------------------------------------

    def get_plot(self, request, passive=False, panel=0):
        """
        Creates one or more plots.

        Creates plot object(s) with compiled attributes
        and directives for drawing one or more visualizations on
        an image.

        Parameters
        ----------
        request : dict
            Options describing the requested plot.
        passive : boolean
            Passive mode. If true, token interpolation is deferred. Returned
            plot directives are descriptive but not executable.
        panel : integer
            Private recursion counter for aggregated plots (do not specify)

        Returns
        -------
        plot : list
            List of one or more Plot objects (see plot.py)

        """

      # Initialize - aggregated plots (i.e. multiple panels) are
      # specified in configuration using the following parameters:
      #
      # /theme/plot/field/layout
      # /theme/plot/field/panels

        theme     = self.name
        field     = request['field']
        path      = [theme, 'plot', field]
        layout    = self.config(path+['layout'], {})
        panels    = self.config(path+['panels'], [{}])

      # End recursion when panels are exhausted.

        if panel+1 > len(panels): return []

      # Update the request for each panel by overriding the 
      # default request with configured settings.

        req = copy.copy(request)
        req.update(layout)
        req.update(panels[panel])
        req.update(self.make_dt(req))
        req['panel'] = panel

      # Create the plot object.

        plot    = Plot(req, self, passive, **request)
        handle  = plot.handle
        layers  = self.plot_init(plot)
        theme   = self.name
        lpath   = [theme,'layer']

        plot.plot_map()

      # Compile directives for each layer of the plot.

        for layer in layers:

            kwargs = {}

            name   = 'plot_' + self.config(lpath + [layer,'gxout'],'contour')
            method = self.config(lpath + [layer,'method'], None)

            if method:
                f = getattr(self, method)
                if f: kwargs.update(f(plot,layer))

            f = getattr(self, name)
            if f: f(plot,layer,**kwargs)

      # Finalize the plot and return the object.

        plot.plot_labels()
        plot.plot_shapes()
        plot.plot_logos()
        plot.cmd('draw grid')
        plot.execute()

        return [plot] + self.get_plot(request, passive, panel+1)

#------------------------------------------------------------------------------

    def plot_init(self, plot):

        handle  = plot.handle
        request = plot.request

        theme      = self.name
        time       = request['time_dt']
        stream     = request['stream']
        field      = request['field']
        ftime      = request.get('fcst_dt',None)

#       region             = request['region']
#       path               = ['region', region]
#       handle.xlab        = self.config(path + ['xlab'],  'on')
#       handle.ylab        = self.config(path + ['ylab'],  'on')

        path               = ['stream',stream]
        handle.model       = self.config(path + ['description'])
        handle.institution = self.config(path + ['institution'])
        handle.subtitle    = '$model|$institution'

        handle.font        = self.config.get('font','')
        handle.mfont       = self.config.get('mfont','')
        handle.bfont       = self.config.get('bfont','')
        handle.grid        = self.config.get('grid','off')
        handle.labsiz      = self.config.get('label_size', '0.14')
        handle.cbar        = self.config.get('colorbar','cbart')

        path = [theme]
        handle.font        = self.config(path + ['font'],  handle.font)
        handle.mfont       = self.config(path + ['mfont'], handle.mfont)
        handle.bfont       = self.config(path + ['bfont'], handle.bfont)
        handle.grid        = self.config(path + ['grid'],  handle.grid)
        handle.labsiz      = self.config(path + ['label_size'], handle.labsiz)

        path               = [theme, 'plot', field]
        handle.font        = self.config(path + ['font'],  handle.font)
        handle.mfont       = self.config(path + ['mfont'], handle.mfont)
        handle.bfont       = self.config(path + ['bfont'], handle.bfont)
        handle.grid        = self.config(path + ['grid'],  handle.grid)
        handle.labsiz      = self.config(path + ['label_size'], handle.labsiz)
        handle.parea       = self.config(path + ['parea'],'off')
        lights             = self.config(path + ['lights'],'on')
        layers             = self.config(path + ['layers'],[])
        handle.title       = self.config(path + ['title'],'')
        handle.label       = self.config(path + ['label'],'')
        handle.tag         = self.config(path + ['tag'],'')

        handle.regular     = 13
        handle.fixed       = 14
        handle.bold        = 15

#       path               = ['annotate']
#       tm_valid           = self.config(path + ['tm_valid'],'')
#       tm_verif           = self.config(path + ['tm_verif'],'')
#       tm_start           = self.config(path + ['tm_start'],'')
#       tm_string          = self.config(path + ['tm_string'],'')

        path               = ['annotate']
        adict              = self.config(path, {})
        tm_valid           = plot.get_attr(adict, 'tm_valid', '')
        tm_verif           = plot.get_attr(adict, 'tm_verif', '')
        tm_start           = plot.get_attr(adict, 'tm_start', '')
        tm_string          = plot.get_attr(adict, 'tm_string', '')
        tm_alt             = plot.get_attr(adict, 'tm_alt', '')

        parea              = request.get('parea', None)
        labsiz             = request.get('label_size', None)

        if parea:  handle.parea   = parea
        if labsiz: handle.labsiz = labsiz
        handle.xlabsiz = handle.labsiz.split()[0]
        handle.ylabsiz = handle.labsiz.split()[-1]

        handle.background = 1
        if lights == 'off': handle.background = 0
        if request.get('lights_off', 0): handle.background = 0

        handle.month     = str(time.month)
        handle.mm1       = str(time.month-1)
        handle.mm2       = str(time.month-2)
        if time.month-1 <= 0: handle.mm1 = str(time.month-1+12)
        if time.month-2 <= 0: handle.mm2 = str(time.month-2+12)

        handle.jday      = time.strftime("%j")
        handle.doy       = handle.jday

        julian_day = int(handle.jday)
        if calendar.isleap(time.year) and julian_day > 59:
            handle.doy = str(julian_day - 1)

        days5            = dt.timedelta(days=5)
        days10           = dt.timedelta(days=10)

        if ftime is None:
            handle.tau       = 0
            handle.tm_start  = time.strftime(tm_start)
            handle.tm_verif  = time.strftime(tm_verif)
            handle.tm_alt    = time.strftime(tm_alt)
            handle.tm_string = handle.tm_verif
            handle.cycle     = 'Analysis'
            handle.tbeg      = time.strftime("%H:%Mz%d%b%Y")
            handle.t5day     = (time+days5).strftime("%H:%Mz%d%b%Y")
            handle.t10day    = (time+days10).strftime("%H:%Mz%d%b%Y")
            handle.lead_month = 0
        else:
            tau              = time - ftime
            hour             = int(tau.total_seconds() / 3600)
            handle.tau       = "%03d"%(hour,)
            handle.tm_valid  = time.strftime(tm_valid)
            handle.tm_alt    = time.strftime(tm_alt)
            handle.tm_start  = ftime.strftime(tm_start)
            handle.tm_string = tm_string
            handle.tbeg      = ftime.strftime("%H:%Mz%d%b%Y")
            handle.t5day     = (ftime+days5).strftime("%H:%Mz%d%b%Y")
            handle.t10day    = (ftime+days10).strftime("%H:%Mz%d%b%Y")

            handle.cycle     = 'Forecast'
            if hour <= 0: handle.cycle = 'Analysis'

          # Lead month calculation for S2S applications.
          # ============================================

            handle.lead_month = time.month - ftime.month + 1
            if handle.lead_month <= 0:
                handle.lead_month += 12
        # Add contextual parameters

        self.add_cf_context(plot)

        plot.cmd("""
          &INIT
          clear
          set xlab on
          set ylab on
          set parea $parea
          set dfile $#
          set font 13 file $font
          set font 14 file $mfont
          set font 15 file $bfont
          set font 13
          set grid $grid
          set background $background
          set cthick 5
          set clopts -1 5 .12
          set xlopts 1 3 $xlabsiz
          set ylopts 1 3 $ylabsiz
          &END
          """
        )

        return layers

#------------------------------------------------------------------------------

    def add_cf_context(self, plot):
        """
        Adds contextual parameters for GEOS CF instances

        This method defines parameters needed for configuring
        GEOS CF instances; especially those parameters needed for
        daily anomaly calculations.

        Parameters
        ----------
        plot : Plot
            Plot object
        plot.handle.doy1 : string
            lower bound day of year for interpolating daily means
        plot.handle.doy2 : string
            upper bound day of year for interpolating daily means
        plot.handle.r1 : string
            lower bound interpolation factor (see notes)
        plot.handle.r2 : string
            upper bound interpolation factor (see notes)
        plot.handle.tm_begin : string
            beginning of daily mean period in user specified format (see note.5)
        plot.handle.tm_end : string
            end of daily mean period in user specified format (see note.5)

        Notes
        -----
        (1) Interpolation for daily means: val = r1*data(doy1) + r2*data(doy2)
        (2) Daily long-term means are assumed to be from 00Z to 23Z.
        (3) A request at hour 12Z (for example) will interpolate between the
            current day and the next day since the daily mean will be
            calculated from 12Z to 11Z of the next day.
        (4) The day of year (doy) is an integer index. Long-term mean files
            are expected to have 365 day entries. For that reason, leap-day is
            set to 59 to keep all other day indices the same for leap years.
        (5) Beginning and ending time formats are defined in the "annotate"
            block of the configuration. For example:
            tm_begin: %Y-%m-%d %HZ
            tm_end: %Y-%m-%d %HZ

        Returns
        -------
        None
            No return value

        """

        handle  = plot.handle
        request = plot.request

        time = request['time_dt']

        # Add interpolation factors for deriving the daily long-term mean
        # (see notes).

        dt1 = time.strftime("%Y%m%d")
        dt2 = (time + dt.timedelta(days=1)).strftime("%Y%m%d")

        dt1 = dt.fromiso(dt1)
        dt2 = dt.fromiso(dt2)

        r1 = (dt2 - time).total_seconds() / (dt2 - dt1).total_seconds()
        r2 = (time - dt1).total_seconds() / (dt2 - dt1).total_seconds()

        doy1 = dt1.timetuple().tm_yday
        doy2 = dt2.timetuple().tm_yday

        if calendar.isleap(dt1.year) and doy1 > 59:
            doy1 -= 1
        if calendar.isleap(dt2.year) and doy2 > 59:
            doy2 -= 1

        handle.doy1 = str(doy1)
        handle.doy2 = str(doy2)
        handle.r1 = str(r1)
        handle.r2 = str(r2)

        # Add start and end date/time for the daily mean. End time is
        # assumed to be equal to the start time plus 23 hours.

        path = ['annotate']
        adict = self.config(path, {})
        tm_begin = plot.get_attr(adict, 'tm_begin', '')
        tm_end = plot.get_attr(adict, 'tm_end', '')

        handle.tm_begin = time.strftime(tm_begin)
        handle.tm_end = (time + dt.timedelta(hours=24)).strftime(tm_end)

#------------------------------------------------------------------------------

    def plot_vort(self, plot, name, **kwargs):

        handle  = plot.handle
        request = plot.request
        region  = request['region']

        layer   = plot.get_layer(name)
        if not layer: return

        expr = plot.get_attr(layer,'expr')
        map  = plot.get_map(request)[0]

        lat = [ float(lat) for lat in map['lat'].split() if lat != ' ' ]
        factor  = str(1.0)
        if lat[-1] <= 0.0:
            factor = str(-1.0)

        expr = '(' + expr + ')' + '*' + factor

        self.plot_shaded(plot, name, expr=expr)

#------------------------------------------------------------------------------

    def plot_contour(self, plot, name, **kwargs):

        handle  = plot.handle
        request = plot.request
        time    = request['time_dt']
        field   = request['field']
        theme   = self.name

        layer   = plot.get_layer(name)
        if not layer: return

        zorder = plot.get_attr(layer,'zorder','-1')
        kwargs.update(plot.get_vars(layer))
        kwargs['layer_name'] = name
        kwargs['zorder'] = zorder

        path = [theme, 'plot', field]
        handle.lat     = self.config(path+['lat'],'--auto')
        handle.lon     = self.config(path+['lon'],'--auto')
        handle.lev     = self.config(path+['lev'],'--auto')
        handle.time    = self.config(path+['time'],'--auto')
        handle.x       = self.config(path+['x'],'--auto')
        handle.y       = self.config(path+['y'],'--auto')
        handle.z       = self.config(path+['z'],'--auto')
        handle.t       = self.config(path+['t'],'--auto')
        handle.xaxis   = self.config(path+['xaxis'],'--auto')
        handle.slice   = self.config(path+['slice'],'')
        handle.zlog    = self.config(path+['zlog'],'--auto')
        handle.label   = self.config(path+['label'],'')
        handle.tag     = self.config(path+['tag'],'')
#       handle.ylab    = self.config(path+['ylab'],'--auto')

        self.set_coords(plot)

        handle.gxout    = kwargs.get('gxout', 'contour')
        handle.gtime    = time.strftime("%H:%Mz%d%b%Y")
        handle.cint     = plot.get_attr(layer,'cint','--auto')
        handle.clevs    = plot.get_attr(layer,'clevs','--auto')
        handle.expr     = plot.get_attr(layer,'expr',handle._STACK_)
        handle.csmooth  = plot.get_attr(layer,'csmooth','off')
        handle.clab     = plot.get_attr(layer,'clab','on')
        handle.ccolor   = plot.get_attr(layer,'ccolor','0 0 0')
        handle.cthick   = plot.get_attr(layer,'cthick','5')
        handle.cstyle   = plot.get_attr(layer,'cstyle','1')
        handle.clcolor  = plot.get_attr(layer,'clcolor',handle.ccolor)
        handle.clthick  = plot.get_attr(layer,'clthick','5')
        handle.clsize   = plot.get_attr(layer,'clsize','0.09')
        handle.clsize   = self.config.get('clsize', handle.clsize)
        handle.clskip   = plot.get_attr(layer,'clskip','1')
        handle.cmark    = plot.get_attr(layer,'cmark','--auto')
        handle.cmin     = plot.get_attr(layer,'cmin','--auto')
        handle.cmax     = plot.get_attr(layer,'cmax','--auto')
        handle.mask     = plot.get_attr(layer,'mask','--auto')
        handle.vrange   = plot.get_attr(layer,'vrange','--auto')
        handle.x        = plot.get_attr(layer,'x',handle.x)
        handle.y        = plot.get_attr(layer,'y',handle.y)
        handle.z        = plot.get_attr(layer,'z',handle.z)
        handle.t        = plot.get_attr(layer,'t',handle.t)
        handle.xaxis    = plot.get_attr(layer,'xaxis',handle.xaxis)
        handle.ylpos    = plot.get_attr(layer,'ylpos','--auto')
        handle.ylint    = plot.get_attr(layer,'ylint','--auto')
        handle.ylopts   = plot.get_attr(layer,'ylopts','--auto')
        handle.log1d    = plot.get_attr(layer,'log1d','--auto')
        handle.baropts  = plot.get_attr(layer,'baropts','--auto')
        handle.ytitle   = plot.get_attr(layer,'ylab','--auto')

        eloop = plot.get_attr(layer,'eloop', 1)
        e1    = int(str(eloop).split()[0])
        e2    = int(str(eloop).split()[-1])

        plot.cmd("""
          set dfile $#
          set time $gtime
          set time $time
          set lev $level
          set lev $lev
          set lat $lat
          set lon $lon
          set SLICE $slice
          set zlog $zlog
          set x $x
          set y $y
          set z $z
          set t $t
          set xaxis $xaxis
          set ylpos $ylpos
          set ylint $ylint
          set ylopts $ylopts
          set grads off
          set gxout $gxout
        """, **kwargs
        )

        cbar = None

        for e in range(e1, e2+1):

            handle.name = plot.get_name()
            plot.cmd("set e %d"%(e,), **kwargs)

            cbar = self.set_shade(plot, layer, zorder)

            if not cbar:

                plot.cmd("""
                  set cint $cint
                  set cmin $cmin
                  set cmax $cmax
                  set clevs $clevs
                  set rgb $* $ccolor
                  set ccolor $*
                  set rgb $* $clcolor
                  set line $* $cstyle $cthick
                  set clopts $* $clthick $clsize
                  set baropts $baropts
                """, **kwargs
                )

            plot.cmd("""
              set vrange $vrange
              set log1d $log1d
              set csmooth $csmooth
              set cmark $cmark
              set cthick $cthick
              set cstyle $cstyle
              set clab $clab
              set clskip $clskip
              define $name = $expr
              define $name = maskout($name,$name-$mask)
              display $name
              draw ylab $ytitle
            """, **kwargs
            )

        handle._STACK_ = handle.name
        if cbar: plot.cmd('draw cbar ' + cbar, **kwargs)

#------------------------------------------------------------------------------

    def plot_grid(self, plot, name, **kwargs):

        handle  = plot.handle
        request = plot.request
        time    = request['time_dt']
        field   = request['field']
        theme   = self.name

        layer   = plot.get_layer(name)
        if not layer: return

        kwargs.update(plot.get_vars(layer))
        kwargs['layer_name'] = name
        kwargs['zorder'] = plot.get_attr(layer,'zorder','0')

        path = [theme, 'plot', field]
        handle.lat     = self.config(path+['lat'],'--auto')
        handle.lon     = self.config(path+['lon'],'--auto')
        handle.lev     = self.config(path+['lev'],'--auto')
        handle.time    = self.config(path+['time'],'--auto')
        handle.x       = self.config(path+['x'],'--auto')
        handle.y       = self.config(path+['y'],'--auto')
        handle.z       = self.config(path+['z'],'--auto')
        handle.t       = self.config(path+['t'],'--auto')
        handle.xaxis   = self.config(path+['xaxis'],'--auto')
        handle.slice   = self.config(path+['slice'],'')
        handle.zlog    = self.config(path+['zlog'],'--auto')

        self.set_coords(plot)

        handle.gtime    = time.strftime("%H:%Mz%d%b%Y")
        handle.name     = plot.get_name()
        handle.expr     = plot.get_attr(layer,'expr','')
        handle.ccolor   = plot.get_attr(layer,'ccolor','0 0 0')
        handle.cthick   = plot.get_attr(layer,'cthick','5')
        handle.cstyle   = plot.get_attr(layer,'cstyle','1')
        handle.cmark    = plot.get_attr(layer,'cmark','0')
        handle.digsiz   = plot.get_attr(layer,'digsiz','--auto')
        handle.gridln   = plot.get_attr(layer,'gridln','--auto')
        handle.skip     = plot.get_attr(layer,'skip','--auto')
        handle.mask     = plot.get_attr(layer,'mask','--auto')
        handle.vrange   = plot.get_attr(layer,'vrange','--auto')
        handle.x        = plot.get_attr(layer,'x',handle.x)
        handle.y        = plot.get_attr(layer,'y',handle.y)
        handle.z        = plot.get_attr(layer,'z',handle.z)
        handle.t        = plot.get_attr(layer,'t',handle.t)
        handle.xaxis    = plot.get_attr(layer,'xaxis',handle.xaxis)
        handle.log1d    = plot.get_attr(layer,'log1d','--auto')

        plot.cmd("""
          set dfile $#
          set time $gtime
          set time $time
          set lev $level
          set lev $lev
          set lat $lat
          set lon $lon
          set x $x
          set y $y
          set t $t
          set xaxis $xaxis
          set SLICE $slice
          set zlog $zlog
          set z $z
          set grads off
          set gxout grid
          set vrange $vrange
          set log1d $log1d
          set rgb $* $ccolor
          set ccolor $*
          set line $*
          set cthick $cthick
          set cstyle $cstyle
          set cmark $cmark
          set digsiz $digsiz
          set gridln $gridln
          define $name = $expr
          define $name = maskout($name,$name-$mask)
          display $name
        """, **kwargs
        )

#------------------------------------------------------------------------------

    def plot_shaded(self, plot, name, **kwargs):

        handle  = plot.handle
        request = plot.request
        time    = request['time_dt']
        field   = request['field']
        theme   = self.name

        layer   = plot.get_layer(name)
        if not layer: return

        kwargs.update(plot.get_vars(layer))
        kwargs['layer_name'] = name

        zorder = plot.get_attr(layer,'zorder','0')
        kwargs['zorder'] = zorder

        path = [theme, 'plot', field]
        handle.lat     = self.config(path+['lat'],'--auto')
        handle.lon     = self.config(path+['lon'],'--auto')
        handle.lev     = self.config(path+['lev'],'--auto')
        handle.time    = self.config(path+['time'],'--auto')
        handle.x       = self.config(path+['x'],'--auto')
        handle.y       = self.config(path+['y'],'--auto')
        handle.z       = self.config(path+['z'],'--auto')
        handle.t       = self.config(path+['t'],'--auto')
        handle.xaxis   = self.config(path+['xaxis'],'--auto')
        handle.slice   = self.config(path+['slice'],'')
        handle.zlog    = self.config(path+['zlog'],'--auto')

        self.set_coords(plot)

        handle.gxout   = kwargs.get('gxout', 'shaded') # SJR 20240724 updated shaded to shade1
        handle.gtime   = time.strftime("%H:%Mz%d%b%Y")
        handle.name    = plot.get_name()
        handle.expr    = plot.get_attr(layer,'expr',handle._STACK_)
        handle.csmooth = plot.get_attr(layer,'csmooth','off')
        handle.mask    = plot.get_attr(layer,'mask','--auto')
        handle.vrange  = plot.get_attr(layer,'vrange','--auto')
        handle.x       = plot.get_attr(layer,'x',handle.x)
        handle.y       = plot.get_attr(layer,'y',handle.y)
        handle.z       = plot.get_attr(layer,'z',handle.z)
        handle.t       = plot.get_attr(layer,'t',handle.t)
        handle.xaxis   = plot.get_attr(layer,'xaxis',handle.xaxis)
        handle.log1d   = plot.get_attr(layer,'log1d','--auto')

        handle.cbar    = self.set_shade(plot, layer, zorder)

        plot.cmd("""
          set dfile $#
          set time $gtime
          set time $time
          set lev $level
          set lev $lev
          set lat $lat
          set lon $lon
          set x $x
          set y $y
          set t $t
          set xaxis $xaxis
          set SLICE $slice
          set zlog $zlog
          set z $z
          set datawarn off
          set gxout $gxout
          set vrange $vrange
          set log1d $log1d
          set csmooth $csmooth
          set grads off
          define $name = $expr
          define $name = maskout($name,$name-$mask)
          d $name
          draw cbar $cbar
        """, **kwargs
        )

        handle._STACK_ = handle.name

#------------------------------------------------------------------------------

    def plot_grfill(self, plot, name, **kwargs):
        kwargs['gxout'] = 'grfill'
        self.plot_shaded(plot, name, **kwargs)

#------------------------------------------------------------------------------

    def plot_bar(self, plot, name, **kwargs):
        kwargs['gxout'] = 'bar'
        self.plot_contour(plot, name, **kwargs)

#------------------------------------------------------------------------------

    def plot_line(self, plot, name, **kwargs):
        kwargs['gxout'] = 'line'
        self.plot_contour(plot, name, **kwargs)

#------------------------------------------------------------------------------

    def plot_barb(self, plot, name, **kwargs):

        handle  = plot.handle
        request = plot.request
        time    = request['time_dt']
        field   = request['field']
        theme   = self.name

        layer   = plot.get_layer(name)
        if not layer: return 

        kwargs.update(plot.get_vars(layer))
        kwargs['layer_name'] = name
        kwargs['zorder'] = plot.get_attr(layer,'zorder','-1')

        path = [theme, 'plot', field]
        handle.lat     = self.config(path+['lat'],'--auto')
        handle.lon     = self.config(path+['lon'],'--auto')
        handle.lev     = self.config(path+['lev'],'--auto')
        handle.time    = self.config(path+['time'],'--auto')
        handle.x       = self.config(path+['x'],'--auto')
        handle.y       = self.config(path+['y'],'--auto')
        handle.z       = self.config(path+['z'],'--auto')
        handle.t       = self.config(path+['t'],'--auto')
        handle.xaxis   = self.config(path+['xaxis'],'--auto')
        handle.slice   = self.config(path+['slice'],'')
        handle.zlog    = self.config(path+['zlog'],'--auto')

        self.set_coords(plot)

        handle.gtime  = time.strftime("%H:%Mz%d%b%Y")
        handle.uname  = plot.get_name()
        handle.vname  = plot.get_name()
        handle.uexpr  = plot.get_attr(layer,'uexpr','')
        handle.vexpr  = plot.get_attr(layer,'vexpr','')
        handle.size   = plot.get_attr(layer,'size','--auto')
        handle.ccolor = plot.get_attr(layer,'ccolor','0 0 0')
        handle.cthick = plot.get_attr(layer,'cthick','5')
        handle.x      = plot.get_attr(layer,'x',handle.x)
        handle.y      = plot.get_attr(layer,'y',handle.y)
        handle.z      = plot.get_attr(layer,'z',handle.z)
        handle.t      = plot.get_attr(layer,'t',handle.t)
        handle.xaxis  = plot.get_attr(layer,'xaxis',handle.xaxis)

        skip          = plot.get_attr(layer,'skip',None)

        if skip:
            skip = [v for v in str(skip).split() if v != ' ']
            handle.xskip, handle.yskip = tuple(skip + skip)[0:2]
        else:
            handle.xskip, handle.yskip = plot.get_skip(handle.uexpr)

        plot.cmd("""
          set dfile $#
          set time $gtime
          set time $time
          set lev $level
          set lev $lev
          set lat $lat
          set lon $lon
          set x $x
          set y $y
          set z $z
          set t $t
          set xaxis $xaxis
          set SLICE $slice
          set zlog $zlog
          set datawarn off
          set gxout barb
          set grads off
          set rgb $* $ccolor
          set ccolor $*
          set cthick $cthick
          set digsiz $size
          define $uname = $uexpr
          define $vname = $vexpr
          d skip($uname,$xskip,$yskip);$vname
        """, **kwargs
        )

#------------------------------------------------------------------------------

    def plot_vector(self, plot, name, **kwargs):

        handle  = plot.handle
        request = plot.request
        time    = request['time_dt']
        field   = request['field']
        theme   = self.name

        layer   = plot.get_layer(name)
        if not layer: return

        kwargs.update(plot.get_vars(layer))
        kwargs['layer_name'] = name
        kwargs['zorder'] = plot.get_attr(layer,'zorder','-1')

        path = [theme, 'plot', field]
        handle.lat     = self.config(path+['lat'],'--auto')
        handle.lon     = self.config(path+['lon'],'--auto')
        handle.lev     = self.config(path+['lev'],'--auto')
        handle.time    = self.config(path+['time'],'--auto')
        handle.x       = self.config(path+['x'],'--auto')
        handle.y       = self.config(path+['y'],'--auto')
        handle.z       = self.config(path+['z'],'--auto')
        handle.t       = self.config(path+['t'],'--auto')
        handle.xaxis   = self.config(path+['xaxis'],'--auto')
        handle.slice   = self.config(path+['slice'],'')
        handle.zlog    = self.config(path+['zlog'],'--auto')

        self.set_coords(plot)

        handle.gtime  = time.strftime("%H:%Mz%d%b%Y")
        handle.uname  = plot.get_name()
        handle.vname  = plot.get_name()
        handle.uexpr  = plot.get_attr(layer,'uexpr','')
        handle.vexpr  = plot.get_attr(layer,'vexpr','')
        handle.shaft  = plot.get_attr(layer,'arrscl','--auto')
        handle.arrfill= plot.get_attr(layer,'arrfill','--auto')
        handle.pos    = plot.get_attr(layer,'arrpos','Lr')
        handle.arrow  = plot.get_attr(layer,'arrowhead','--auto')
        handle.arrlab = plot.get_attr(layer,'arrlab','on')
        handle.ccolor = plot.get_attr(layer,'ccolor','0 0 0')
        handle.cthick = plot.get_attr(layer,'cthick','5')
        handle.x      = plot.get_attr(layer,'x',handle.x)
        handle.y      = plot.get_attr(layer,'y',handle.y)
        handle.z      = plot.get_attr(layer,'z',handle.z)
        handle.t      = plot.get_attr(layer,'t',handle.t)
        handle.xaxis  = plot.get_attr(layer,'xaxis',handle.xaxis)

        skip          = plot.get_attr(layer,'skip',None)

        if skip:
            skip = [v for v in str(skip).split() if v != ' ']
            handle.xskip, handle.yskip = tuple(skip + skip)[0:2]
        else:
            handle.xskip, handle.yskip = plot.get_skip(handle.uexpr)

        plot.cmd("""
          set dfile $#
          set time $gtime
          set time $time
          set lev $level
          set lev $lev
          set lat $lat
          set lon $lon
          set x $x
          set y $y
          set z $z
          set t $t
          set xaxis $xaxis
          set SLICE $slice
          set zlog $zlog
          set datawarn off
          set gxout vector
          set grads off
          set rgb $* $ccolor
          set ccolor $*
          set line $*
          set ARRFILL $arrfill
          set cthick $cthick
          set arrscl $shaft
          set arrowhead $arrow
          set arrlab off
          define $uname = $uexpr
          define $vname = $vexpr
          d skip($uname,$xskip,$yskip);$vname
          draw arrow $arrlab $shaft $pos
        """, **kwargs
        )

#------------------------------------------------------------------------------

    def plot_stream(self, plot, name, **kwargs):

        handle  = plot.handle
        request = plot.request
        time    = request['time_dt']
        field   = request['field']
        theme   = self.name

        layer   = plot.get_layer(name)
        if not layer: return

        kwargs.update(plot.get_vars(layer))
        kwargs['layer_name'] = name

        zorder = plot.get_attr(layer,'zorder','-1')
        kwargs['zorder'] = zorder

        path = [theme, 'plot', field]
        handle.lat     = self.config(path+['lat'],'--auto')
        handle.lon     = self.config(path+['lon'],'--auto')
        handle.lev     = self.config(path+['lev'],'--auto')
        handle.time    = self.config(path+['time'],'--auto')
        handle.x       = self.config(path+['x'],'--auto')
        handle.y       = self.config(path+['y'],'--auto')
        handle.z       = self.config(path+['z'],'--auto')
        handle.t       = self.config(path+['t'],'--auto')
        handle.xaxis   = self.config(path+['xaxis'],'--auto')
        handle.slice   = self.config(path+['slice'],'')
        handle.zlog    = self.config(path+['zlog'],'--auto')

        self.set_coords(plot)

        handle.gtime  = time.strftime("%H:%Mz%d%b%Y")
        handle.uname  = plot.get_name()
        handle.vname  = plot.get_name()
        handle.cname  = plot.get_name()
        handle.uexpr  = plot.get_attr(layer,'uexpr','')
        handle.vexpr  = plot.get_attr(layer,'vexpr','')
        handle.cexpr  = plot.get_attr(layer,'cexpr','')
        handle.strmden= plot.get_attr(layer,'strmden','--auto')
        handle.density= plot.get_attr(layer,'density',handle.strmden)
        handle.ccolor = plot.get_attr(layer,'ccolor','0 0 0')
        handle.cthick = plot.get_attr(layer,'cthick','5')
        handle.x      = plot.get_attr(layer,'x',handle.x)
        handle.y      = plot.get_attr(layer,'y',handle.y)
        handle.z      = plot.get_attr(layer,'z',handle.z)
        handle.t      = plot.get_attr(layer,'t',handle.t)
        handle.xaxis  = plot.get_attr(layer,'xaxis',handle.xaxis)

        handle.cbar   = self.set_shade(plot, layer, zorder)

        plot.cmd("""
          set dfile $#
          set time $gtime
          set time $time
          set lev $level
          set lev $lev
          set lat $lat
          set lon $lon
          set x $x
          set y $y
          set z $z
          set t $t
          set xaxis $xaxis
          set SLICE $slice
          set zlog $zlog
          set datawarn off
          set gxout stream
          set grads off
          set rgb $* $ccolor
          set ccolor $*
          set cthick $cthick
          set strmden $density
          define $uname = $uexpr
          define $vname = $vexpr
        """, **kwargs
        )

        if handle.cexpr:
            plot.cmd("""
              define $cname = $cexpr
              d $uname;$vname;$cname
              draw cbar $cbar
            """, **kwargs
            )
        else:
            plot.cmd('d $uname;$vname', **kwargs)

#------------------------------------------------------------------------------

    def plot_hilo(self, plot, name, **kwargs):

        handle  = plot.handle
        request = plot.request
        time    = request['time_dt']
        field   = request['field']
        theme   = self.name

        layer   = plot.get_layer(name)
        if not layer: return

        kwargs.update(plot.get_vars(layer))
        kwargs['layer_name'] = name
        kwargs['zorder'] = plot.get_attr(layer,'zorder','-1')

        path = [theme, 'plot', field]
        handle.lat     = self.config(path+['lat'],'--auto')
        handle.lon     = self.config(path+['lon'],'--auto')
        handle.lev     = self.config(path+['lev'],'--auto')
        handle.time    = self.config(path+['time'],'--auto')
        handle.gtime   = time.strftime("%H:%Mz%d%b%Y")
        handle.x       = self.config(path+['x'],'--auto')
        handle.y       = self.config(path+['y'],'--auto')
        handle.z       = self.config(path+['z'],'--auto')
        handle.t       = self.config(path+['t'],'--auto')
        handle.name    = plot.get_name()
        handle.expr    = plot.get_attr(layer,'expr','')
        handle.mask    = plot.get_attr(layer,'mask','--auto')

        args = {}
        args['expr']    = handle.name
        args['algo']    = plot.get_attr(layer,'algo','CL')
        args['radius']  = plot.get_attr(layer,'radius','1000')
        args['cint']    = plot.get_attr(layer,'cint','300')
        args['hmark']   = plot.get_attr(layer,'hmark','H')
        args['lmark']   = plot.get_attr(layer,'lmark','L')
        args['color']   = plot.get_attr(layer,'color','0 0 0')

        handle.args     = json.dumps(args)

        plot.cmd("""
          set dfile $#
          set time $gtime
          set time $time
          set lev $level
          set lev $lev
          set lat $lat
          set lon $lon
          set x $x
          set y $y
          set z $z
          set t $t
          set grads off
          define $name = $expr
          define $name = maskout($name,$name-$mask)
          draw hilo $args
        """, **kwargs
        )

#------------------------------------------------------------------------------

    def ice_maker(self, plot, name):

        handle  = plot.handle
        request = plot.request
        time    = request['time_dt']

        layer   = plot.get_layer(name)
        if not layer: return

        handle.gtime  = time.strftime("%H:%Mz%d%b%Y")

        handle.snow        = plot.get_name()
        handle.phis        = plot.get_name()
        handle.thick       = plot.get_name()
        handle.ice         = plot.get_name()
        handle.elevFactor  = plot.get_name()
        handle.thlow       = plot.get_name()

        handle.arg_snow  = plot.get_attr(layer,'snow')
        handle.arg_phis  = plot.get_attr(layer,'phis')
        handle.arg_thick = plot.get_attr(layer,'thick')

        plot.cmd("""
          set dfile $#
          set time $gtime
          set lev $level
          define $phis  = $arg_phis/9.81
          define $snow  = $arg_snow
          define $thick = $arg_thick
          define $elevFactor = (($phis-305.0)/915.0)
          define $thlow = ($elevFactor*600)+5400
          define $thlow = const(maskout($thlow,$elevFactor),5400,'-u')
          define $thlow = const(maskout($thlow,-$elevFactor+1),6000,'-u')
          define $ice   = maskout(maskout($snow,$thick-$thlow),$snow-0.0001)
        """
        )

        return {'expr': handle.ice}

#------------------------------------------------------------------------------

    def vorticity(self, plot, name):

        handle  = plot.handle
        request = plot.request
        time    = request['time_dt']

        layer   = plot.get_layer(name)
        if not layer: return

        map  = plot.get_map(request)[0]
        lat  = [ float(lat) for lat in map['lat'].split() if lat != ' ' ]
        expr = plot.get_attr(layer,'expr')

        handle.gtime     = time.strftime("%H:%Mz%d%b%Y")
        handle.vort      = plot.get_name()
        handle.arg_uwnd  = plot.get_attr(layer,'uwnd')
        handle.arg_vwnd  = plot.get_attr(layer,'vwnd')

        plot.cmd("""
          set dfile $#
          set time $gtime
          set lev $level
        """
        )

        if lat[-1] <= 0.0:
            plot.cmd('define $vort = -hcurl($arg_uwnd,$arg_vwnd)')
        else:
            plot.cmd('define $vort = hcurl($arg_uwnd,$arg_vwnd)')

        return {'expr': expr}

#------------------------------------------------------------------------------

    def tadv(self, plot, name):

        handle  = plot.handle
        request = plot.request
        time    = request['time_dt']

        layer   = plot.get_layer(name)
        if not layer: return

        expr = plot.get_attr(layer,'expr')

        handle.gtime     = time.strftime("%H:%Mz%d%b%Y")
        handle.tadv      = plot.get_name()
        handle.dtx       = plot.get_name()
        handle.dty       = plot.get_name()
        handle.dx        = plot.get_name()
        handle.dy        = plot.get_name()
        handle.arg_uwnd  = plot.get_attr(layer,'uwnd')
        handle.arg_vwnd  = plot.get_attr(layer,'vwnd')
        handle.arg_tmpu  = plot.get_attr(layer,'tmpu')

        plot.cmd("""
          set dfile $#
          set time $gtime
          set lev $level
          define $dtx   = cdiff($arg_tmpu,x)
          define $dty   = cdiff($arg_tmpu,y)
          define $dx    = cdiff(lon,x)*3.1416/180
          define $dy    = cdiff(lat,y)*3.1416/180
          define $tadv  = -1*( ($arg_uwnd*$dtx)/(cos(lat*3.1416/180)*$dx) + $arg_vwnd*$dty/$dy )/6.37e+6
        """
        )

        return {'expr': expr}

#------------------------------------------------------------------------------

    def shtorh(self, plot, name):

        handle  = plot.handle
        request = plot.request
        time    = request['time_dt']

        layer   = plot.get_layer(name)
        if not layer: return

        expr = plot.get_attr(layer,'expr')

        handle.gtime     = time.strftime("%H:%Mz%d%b%Y")
        handle.rh        = plot.get_name()
        handle.a         = plot.get_name()
        handle.b         = plot.get_name()
        handle.es        = plot.get_name()
        handle.qs        = plot.get_name()
        handle.rhum      = plot.get_name()
        handle.arg_temp  = plot.get_attr(layer,'temp')
        handle.arg_shum  = plot.get_attr(layer,'shum')
        handle.arg_pres  = plot.get_attr(layer,'pres')

        plot.cmd("""
          set dfile $#
          set time $gtime
          set lev $level
          define $a     = 21.874
          define $b     = 7.66
          define $a     = maskout($a, -$arg_temp+273.16)
          define $b     = maskout($b, -$arg_temp+273.16)
          define $a     = const($a, 17.269, "-u")
          define $b     = const($b, 35.86, "-u")
          define $es    = 6.11*exp($a*($arg_temp-273.16)/($arg_temp-$b))
          define $qs    = 0.622*($es/($arg_pres-0.378*$es))
          define $rhum  = ($arg_shum/$qs)*100
        """
        )

        return {'expr': expr}

#------------------------------------------------------------------------------

    def set_shade(self, plot, layer, zorder=0):

        clevs  = plot.get_attr(layer,'clevs',None)
        cmin   = plot.get_attr(layer,'cmin',None)
        cmax   = plot.get_attr(layer,'cmax',None)
        cint   = plot.get_attr(layer,'cint',None)
        nsub   = plot.get_attr(layer,'nsub',1)
        type   = plot.get_attr(layer,'type','linear')
        cbar   = plot.get_attr(layer,'cbar', None)

        if not cbar: return

        rgba     = self.config(['attribute','colorbar',cbar], cbar)

        tklabsiz = plot.request.get('tick_label_size', None)

        cbopts   = { 'scale':    plot.get_attr(layer,'scale',None),
                     'reverse':  plot.get_attr(layer,'reverse',None),
                     'inverse':  plot.get_attr(layer,'inverse',None),
                     'alpha':    plot.get_attr(layer,'alpha',None),
                     'sf':       plot.get_attr(layer,'sf','1.0'),
                     'skip':     plot.get_attr(layer,'skip',None),
                     'cblabel':  plot.get_attr(layer,'cbunits',None),
                     'cbpos':    plot.get_attr(layer,'cbpos',None),
                     'tklabsiz': plot.get_attr(layer,'tklabsiz',tklabsiz),
                     'tkscale':  plot.get_attr(layer,'tkscale',None)
                   }

        cbopts   = { k:v for k,v in cbopts.items() if v is not None }

        args     = json.dumps(cbopts)

        clevs = plot.set_clevs(clevs, cmin, cmax, cint)
        plot.set_shade(clevs, rgba, nsub, type, zorder, **cbopts)

        if clevs: return args

        return None

#------------------------------------------------------------------------------

    def set_coords(self, plot):

        handle = plot.handle

        if not handle.slice: return

        coords = [float(v) for v in handle.slice.split()[0:4] if v != ' ']

        handle.lon = '%s %s'%(coords[0], coords[2])
        if coords[0] > coords[2]: handle.lon = '%s %s'%(coords[2], coords[0])

        handle.lat = '%s %s'%(coords[1], coords[3])
        if coords[1] > coords[3]: handle.lat = '%s %s'%(coords[3], coords[1])

#------------------------------------------------------------------------------

    def make_dt(self, request):

        tdict = {}

        for tm_arg in ['fcst_dt', 'time_dt']:

            dattim = request.get(tm_arg, None)
            if not dattim: continue

            if getattr(dattim, 'strftime') or isinstance(dattim,dt.datetime):

                request[tm_arg] = dt.datetime(dattim.year, dattim.month,
                                       dattim.day, dattim.hour, dattim.minute)
                continue

            dattim = str(dattim)

            if len(dattim) <= 8:
                dattim += 'T000000'
            else:
                dattim += '000000'

            tdict[tm_arg] = dt.datetime.strptime(dattim[0:14],'%Y%m%dT%H%M%S')

        shift_dt = request.get('shift_dt', 0)
        if shift_dt: tdict['fcst_dt'] = request['fcst_dt'] + \
                                           dt.timedelta(hours=shift_dt)

        return tdict

#------------------------------------------------------------------------------

    def nlog(self,x,a):
        if a<=0:
            raise ValueError('Expected a>0 but got a=%f'%a)
        return log(a*x+1.0)/log(a+1.0)

#------------------------------------------------------------------------------
    def nexp(self,x,a):
        if a<=0:
            raise ValueError('Expected a>0 but got a=%f'%a)
        return (exp(x * log(a+1.0)) - 1.0) / a

#------------------------------------------------------------------------------

    def log_scale(self, x): return self.nlog(x,10.0)

#------------------------------------------------------------------------------

    def log_scale1(self, x): return self.nlog(x,1.0)
#------------------------------------------------------------------------------

    def exp_scale(self, x): return self.nexp(x,10.0)

#------------------------------------------------------------------------------

    def exp_scale20(self, x): return self.nexp(x,20.0)

#------------------------------------------------------------------------------

    def exp_scale30(self, x): return self.nexp(x,30.0)

#------------------------------------------------------------------------------

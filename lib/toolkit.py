import os
import json
import mydatetime as dt

class Toolkit(object):
  
    def draw(self, plot, shapes):

        for name in shapes:
            shape = getattr(self, name, None)

            if shape is None:
                next

            shape(plot, shapes[name])

    def line(self, plot, lines, **kwargs):

        handle = plot.handle
        handle.line_color = kwargs.get('line_color','255 0 0')
        handle.line_width = kwargs.get('line_width','5')
        handle.line_style = kwargs.get('line_style','1')
        handle.clip       = str(kwargs.get('clip','1'))
        zorder            = kwargs.get('zorder','-1')

        for line in lines:

            if isinstance(lines, dict):

                collection = dict(kwargs)
                collection.update(lines[line])
                self.flight_path(plot, collection['data'], **collection)

            else:

                handle.line = line

                plot.cmd("""
                  set rgb $* $line_color
                  set line $* $line_style $line_width
                  set CLIP $clip
                  draw line $line
                """, zorder=zorder
                )

    def polygon(self, plot, polygons, **kwargs):

        handle = plot.handle
        handle.line_color = kwargs.get('line_color','255 0 0')
        handle.line_width = kwargs.get('line_width','5')
        handle.line_style = kwargs.get('line_style','1')
        zorder            = kwargs.get('zorder','-1')

        for poly in polygons:

            if isinstance(polygons, dict):

                collection = dict(kwargs)
                collection.update(polygons[poly])
                self.polygon(plot, collection['data'], **collection)

            else:

                handle.poly = poly

                plot.cmd("""
                  set rgb $* $line_color
                  set line $* $line_style $line_width
                  draw polyf $poly
                """, zorder=zorder
                )

    def rectangle(self, plot, rectangles, **kwargs):

        handle = plot.handle
        handle.line_color = kwargs.get('line_color','255 0 0')
        handle.fill_color = kwargs.get('fill_color',None)
        handle.line_width = kwargs.get('line_width','5')
        handle.line_style = kwargs.get('line_style','1')
        zorder            = kwargs.get('zorder','-1')

        for rect in rectangles:

            if isinstance(rect, dict):

                collection = dict(kwargs)
                collection.update(rect)
                self.rectangle(plot, collection['data'], **collection)

            elif isinstance(rectangles, dict):

                collection = dict(kwargs)
                collection.update(rectangles[rect])
                self.rectangle(plot, collection['data'], **collection)

            else:

                handle.rect = rect

                if handle.fill_color:
                    plot.cmd("""
                      set rgb $* $fill_color
                      set line $* $line_style $line_width
                      draw recf $rect
                    """, zorder=zorder
                    )

                if handle.line_color:
                    plot.cmd("""
                      set rgb $* $line_color
                      set line $* $line_style $line_width
                      draw rec $rect
                    """, zorder=zorder
                    )

    def string(self, plot, strings, **kwargs):

        handle = plot.handle
        handle.line_color = kwargs.get('line_color','255 255 255')
        handle.line_color = kwargs.get('str_color',handle.line_color)
        handle.line_width = kwargs.get('line_width','5')
        handle.line_width = kwargs.get('str_width',handle.line_width)
        handle.str_size   = kwargs.get('str_size','0.1 0.1')
        handle.position   = kwargs.get('position','c')
        handle.rotation   = kwargs.get('rotation','0')
        handle.font       = '$' + kwargs.get('font','regular')
        handle.clip       = str(kwargs.get('clip','1'))
        zorder            = kwargs.get('zorder','-1')

        for s in strings:

            if isinstance(s, dict):

                collection = dict(kwargs)
                collection.update(s)
                self.string(plot, collection['data'], **collection)

            elif isinstance(strings, dict):

                collection = dict(kwargs)
                collection.update(strings[s])
                self.string(plot, collection['data'], **collection)

            else:

                handle.str = s

                plot.cmd("""
                  set rgb $* $line_color
                  set font $font
                  set CLIP $clip
                  set string $* $position $line_width $rotation
                  set strsiz $str_size
                  draw string $str
                  """, zorder=zorder
                  )

    def mark(self, plot, marks, **kwargs):

        handle = plot.handle
        handle.fill_color   = kwargs.get('fill_color','255 0 0')
        handle.line_color   = kwargs.get('line_color',handle.fill_color)
        handle.line_width   = kwargs.get('line_width','5')
        handle.line_style   = kwargs.get('line_style','1')
        handle.size         = kwargs.get('size','.15')
        handle.type         = kwargs.get('mark','3')
        zorder              = kwargs.get('zorder','-1')
        fname               = kwargs.get('file', None)

        if fname and not isinstance(marks, dict):
            marks = self.read_from_file(fname, **kwargs)

        for mark in marks:

            if isinstance(marks, dict):

                collection = dict(kwargs)
                collection.update(marks[mark])
                self.mark(plot, collection.get('data',None), **collection)

            else:

                handle.mark = mark

                plot.cmd("""
                  set rgb $* $line_color
                  set line $* $line_style $line_width
                  draw mark $type $mark $size
                """, zorder=zorder
                )

    def station_mark(self, plot, marks, **kwargs):

        handle = plot.handle
        handle.line_color   = kwargs.get('line_color','255 0 0')
        handle.fill_color   = kwargs.get('fill_color','255 255 255')
        handle.line_width   = kwargs.get('line_width','5')
        handle.line_style   = kwargs.get('line_style','1')
        handle.inner_size   = kwargs.get('inner_size','.15')
        handle.outer_size   = kwargs.get('outer_size','.20')
        handle.outer_line   = kwargs.get('outer_line','.25')
        zorder              = kwargs.get('zorder','-1')

        if not handle.outer_line:
            handle.outer_line = '--auto'

        mark_type           = kwargs.get('mark_type','3 2')
        mark_type           = [c for c in mark_type if c != ' ']

        handle.mark_type1 = mark_type[0]
        handle.mark_type2 = mark_type[1]

        for mark in marks:

            if isinstance(marks, dict):

                collection = dict(kwargs)
                collection.update(marks[mark])
                self.station_mark(plot, collection['data'], **collection)

            else:

                mark = [s for s in mark.split() if s != ' ']

                handle.navigate = ''
                handle.mark     = ' '.join(mark[0:2])
                if len(mark) > 2: handle.navigate = ' '.join(mark[2:])

                plot.cmd("""
                  set rgb $* $fill_color
                  set line $* $line_style $line_width
                  draw mark $mark_type1 $mark $outer_size
                  set rgb $* $line_color
                  set line $* $line_style $line_width
                  draw mark $mark_type2 $mark $outer_line
                  draw mark $mark_type2 $mark $inner_size $navigate
                """, zorder=zorder
                )

    def symbol(self, plot, symbols, **kwargs):

        handle = plot.handle
        handle.fill_color   = kwargs.get('fill_color','255 0 0')
        handle.line_color   = kwargs.get('line_color',handle.fill_color)
        handle.line_width   = kwargs.get('line_width','5')
        handle.line_style   = kwargs.get('line_style','1')
        handle.size         = kwargs.get('size','0.1')
        handle.type         = kwargs.get('type','3')
        handle.clip         = str(kwargs.get('clip','1'))
        zorder              = kwargs.get('zorder','-1')

        for s in symbols:

          # Case-1: "s" is a dictionary in
          # an array of dictionaries

            if isinstance(s, dict):

                collection = dict(kwargs)
                collection.update(s)
                self.symbol(plot, collection['data'], **collection)

          # Case-2: "s" is a key in a dictionary

            elif isinstance(symbols, dict):

                assert isinstance(symbols[s], dict), \
                    'symbol instance must be a dictionary'

                collection = dict(kwargs)
                collection.update(symbols[s])
                self.symbol(plot, collection['data'], **collection)

          # Case-3: "s" is a location instance

            else:

                handle.location = s

                try:
                    int(handle.type)
                except:

                  # Symbol is an image

                    plot.cmd("""
                      draw symbol $type $location $size
                    """, zorder=zorder
                    )

                else:

                  # Symbol is a weather symbol type.

                    plot.cmd("""
                      set rgb $* $line_color
                      draw wxsym $type $location $size $* $line_width
                    """, zorder=zorder
                    )

    def track(self, plot, tracks, **kwargs):

        handle = plot.handle
        handle.line_color   = kwargs.get('line_color','0 0 0')
        handle.line_width   = kwargs.get('line_width','5')
        handle.line_style   = kwargs.get('line_style','1')
        handle.size         = kwargs.get('size','5')
        handle.symbol       = kwargs.get('symbol','')
        handle.spacing      = kwargs.get('spacing','3')
        handle.position     = kwargs.get('position','c')

        window              = int(kwargs.get('window','500'))
        spacing             = int(kwargs.get('spacing','1')) * 3600
        zorder              = kwargs.get('zorder','-1')

        for t in tracks:

          # Case-1: Looping over an array of dictionaries

            if isinstance(t, dict):

                collection = dict(kwargs)
                collection.update(t)
                self.track(plot, collection['data'], **collection)

          # Case-2: Looping over a dictionary of dictionaries

            elif isinstance(tracks, dict):

                assert isinstance(tracks[t], dict), \
                    'track instance must be a dictionary'

                collection = dict(kwargs)
                collection.update(tracks[t])
                self.track(plot, collection['data'], **collection)

          # Case-3: Track instance

            else:

                plot_dt    = kwargs['time_dt']
                lons       = kwargs['lon'].split()
                wlon, elon = [float(l) for l in lons if l != ''][0:2]
                start_dt   = plot_dt - dt.timedelta(hours=window)
                track_data = self.read_track_data(t, **kwargs)

                for name,record in track_data.items():

                   first_dt = self.track_unpack(record[0])[0]
                   last_dt  = self.track_unpack(record[-1])[0]

                 # Skip the entire feature if the track epoch does
                 # not contain the plot date/time.

                   if first_dt > plot_dt: continue
                   if last_dt  < plot_dt: continue

                   record = [self.track_unpack(r) for r in record]
                   record = self.track_interpolate(record)

                   kwargs['reflon'] = wlon
                   self.track_plot(plot,name,record,start_dt,plot_dt,**kwargs)

                   if (elon - wlon) <= 180.0: continue # Assume no wrap-around

                   kwargs['reflon'] = elon
                   self.track_plot(plot,name,record,start_dt,plot_dt,**kwargs)

    def track_plot(self, plot, name, record, start_dt, end_dt, **kwargs):

        mproj      = kwargs.get('mproj', 'latlon')
        reflon     = kwargs.get('reflon', 0.0)
        spacing    = int(kwargs.get('spacing','1')) * 3600
        lons       = kwargs['lon'].split()
        wlon, elon = [float(l) for l in lons if l != ''][0:2]
        lats       = kwargs['lat'].split()
        slat, nlat = [float(l) for l in lats if l != ''][0:2]

        prev_dt   = None
        locations = []

        for n, rec in enumerate(record):

            time_dt, type, rlat, rlon = rec[0:4]

          # Skip the track location if it is outside
          # the desired time window.

            if time_dt < start_dt: continue
            if time_dt > end_dt: continue

          # Adjust the longitude to be consistent
          # with the region specification.

            rlon   = reflon + self.dirdif(reflon, rlon)
            reflon = rlon

            loc = str(rlon) + ' ' + str(rlat)

            if mproj == 'orthogr' and rlon < wlon: loc = None
            if mproj == 'orthogr' and rlon > elon: loc = None
            if mproj == 'nps' and rlat < slat: loc = None
            if mproj == 'sps' and rlat > slat: loc = None

            locations.append(loc)

            if not loc: continue

          # Plot the symbol at the desired intervals

            if type == 'XX': continue

            if not prev_dt or (time_dt-prev_dt).seconds >= spacing:
                if rlat < 0.0: type += '_SH'
                kwargs['type'] = type
                self.symbol(plot, [locations[-1]], **kwargs)
                prev_dt = time_dt

            if type == 'traj' and rec[4]:
                if n%2 == 0:
                    kwargs['position'] = 'r'
                    self.string(plot, [loc + ' ' + str(rec[4])+'---'], **kwargs)
             #  else:
             #      self.string(plot, [loc + ' ' + '--'], **kwargs)
                if (n+1)%2 == 0:
                    time = str(n*3)
                    kwargs['position'] = 'l'
                    self.string(plot, [loc + ' ' + '--'+time+'h'], **kwargs)

      # Plot the line segments for the track
                       
        for n in range(1,len(locations)):
            if not locations[n-1]: continue
            if not locations[n]: continue
            self.line(plot, [' '.join(locations[n-1:n+1])], **kwargs)

        if type == 'traj':
            return

      # Plot the label for the track

        locations = [loc for loc in locations if loc]
        if not locations: return

        name = name.split('_')[-1]
        loc  = locations[-1]
        self.string(plot, [loc + ' ' + name], **kwargs)

    def read_track_data(self, file, **kwargs):

        time_dt = kwargs['time_dt']
        fcst_dt = kwargs.get('fcst_dt', None)

        if (time_dt):
            file = time_dt.strftime(file)
        if (fcst_dt):
            file = fcst_dt.strftime(file)

        if not os.path.isfile(file):
            return {}

        with open(file, 'r') as f:
            feature = json.load(f)

        return feature

    def dirdif(self, dir1, dir2):

        if dir1 < 0.0: dir1 += 360.0
        if dir2 < 0.0: dir2 += 360.0

        if abs(dir2 - dir1) <= 180.0: return dir2 - dir1
        if dir2 > 180.0: return dir2 - dir1 - 360.0
        return dir2 - dir1 + 360.0

    def track_interpolate(self, record):

        if len(record) < 2: return record
        if record[0][1] == 'traj': return record

        start_dt = record[0][0]
        if start_dt.year < 1980: return record # datetime throws exception

        times = []
        lats  = []
        lons  = []
        symbols = {}

      # Prepare the track data

        for rec in record:

            time_dt, type, rlat, rlon = rec[0:4]

            if not lons: slon = rlon
            rlon = slon + self.dirdif(slon, rlon)

            t = int(round((time_dt - start_dt).total_seconds() / 60.0))

            times.append(t)
            symbols[t] = type
            lats.append(rlat)
            lons.append(rlon)

      # Interpolate to hourly intervals. Include all original
      # track vertices.

        recout = []
        for i in range(0, len(times)-1):

            t1 = times[i]
            t2 = times[i+1]

            for t in range(t1, t2, 180):

                r1 = float(t2 - t) / (t2 - t1)
                r2 = float(t - t1) / (t2 - t1)

                lat = r1 * lats[i] + r2 * lats[i+1]
                lon = r1 * lons[i] + r2 * lons[i+1]
                sym = symbols.get(t, 'XX')
                t_dt = start_dt + dt.timedelta(minutes=t)

                recout.append( (t_dt, sym, lat, lon) )

      # Include the last vertice and return

        recout.append(record[-1])

        return recout    

    def track_unpack(self, record):

        year, month, day, hour, type, lat, lon = record[0:7]
        time_dt = dt.datetime(year, month, day, hour)

        return [time_dt, str(type)] + record[5:]

    def read_from_file(self, fname, **kwargs):

        time_dt = kwargs['time_dt']
        fcst_dt = kwargs.get('fcst_dt', None)

        file = fname

        if (time_dt):
            file = time_dt.strftime(file)
        if (fcst_dt):
            file = fcst_dt.strftime(file)

        if not os.path.isfile(file):
            return []

        with open(file, 'r') as f:
            jdata = json.load(f)

        return jdata

    __call__ = draw

    flight_path = line

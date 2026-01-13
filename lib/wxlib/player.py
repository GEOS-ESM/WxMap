from myutils import parse_duration

class Player(object):

    def __init__(self, playlist, **kwargs):

        self.playlist = dict(playlist)
        self.playlist.update(kwargs)

    def __iter__(self):

        for play in self.iter(self.playlist):

            fields = play.get('fields', [None])
            levels = play.get('levels', [None])
            regions = play.get('regions', [None])

            fcst_dt = play.get('fcst_dt', None)
            start_dt = play.get('start_dt', 'PT0H')
            end_dt = play.get('end_dt', 'PT0H')
            t_deltat = play.get('t_deltat', 'PT1H')
            time_dt = play['time_dt']
            tloop = play.get('tloop', 1)

            if fcst_dt:
                fcst_dt = time_dt + parse_duration(fcst_dt)

            start_dt = time_dt + parse_duration(start_dt)
            end_dt = time_dt + parse_duration(end_dt)
            t_deltat = parse_duration(t_deltat)
            play.update({'start_dt': start_dt,
                         'end_dt': end_dt,
                         'fcst_dt': fcst_dt,
                         't_deltat': t_deltat})

            for field in fields:
                for level in levels:
                    for region in regions:

                        play.update({'field': field,
                                     'level': level,
                                     'region': region})

                        if not tloop:
                            yield play
                            continue

                        t = start_dt
                        while t <= end_dt:
                            play.update({'start_dt': t,'end_dt': t})
                            yield dict(play)
                            t += t_deltat

    def iter(self, playlist, **kwargs):

        d = dict(kwargs)
        d.update({k:v for k,v in playlist.items() if not isinstance(v,dict)})

        children = [v for v in playlist.values() if isinstance(v,dict)]

        if not children:
            return [d]

        leaves = []
        for child in children:
            leaves += self.iter(child, **d)

        return leaves

def get_plays(config, play):

    plays = []
    playlist = config.get(play, {})
    if isinstance(playlist, list):
        for play in playlist:
            plays += get_plays(config, play)
    else:
        return [playlist]

    return plays

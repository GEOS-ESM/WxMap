import sys
import argparse

import mydatetime as dt


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rc', metavar='RESOURCE', default='',
        help='Name of resource file (default: %(default)s)'
    )
    parser.add_argument(
        '--config', metavar='CONFIG', default=[], action='append',
        help='Name of configuration file or directory'
    )
    parser.add_argument(
        '--reset', metavar='RESET', default=[], action='append',
        help='Name of resources to reset'
    )
    parser.add_argument(
        '--motif', metavar='MOTIF', default=[], action='append',
        help='Name of motif file or directory'
    )
    parser.add_argument(
        '--theme', metavar='THEME', default=[], action='append',
        help='Name of configuration file or directory referencing a theme'
    )
    parser.add_argument(
        '-g', '--geometry', metavar='GEOMETRY', default='1024x768',
        help='Image size in pixels (default: %(default)s)'
    )
    parser.add_argument(
        '-r', '--region', metavar='REGION', default='',
        help='Name of region (default: %(default)s)'
    )
    parser.add_argument(
        '-f', '--field', metavar='FIELD', default='',
        help='Name of field (default: %(default)s)'
    )
    parser.add_argument(
        '-p', '--plot', metavar='PLOT', default='',
        help='Name of field (default: %(default)s)'
    )
    parser.add_argument(
        '-l', '--level', metavar='LEVEL', default='',
        help='Pressure level (default: %(default)s)'
    )
    parser.add_argument(
        '-s', '--stream', metavar='STREAM', default='G5FPFC',
        help='Name of stream (default: %(default)s)'
    )
    parser.add_argument(
        '-c', '--collection', metavar='COLLECTION', default='',
        help='Name of collection (default: %(default)s)'
    )
    parser.add_argument(
        '--layer', metavar='LAYER', default='',
        help='Name of layer (default: %(default)s)'
    )
    parser.add_argument(
        '-t', '--time_dt', metavar='YYYYMMDDTHHMMSS', default=None,
        help='Time in ISO format'
    )
    parser.add_argument(
        '--fcst_dt', metavar='YYYYMMDDTHHMMSS', default=None,
        help='Forecast start time in ISO format'
    )
    parser.add_argument(
        '--start_dt', metavar='YYYYMMDDTHHMMSS', default=None,
        help='Start time in ISO format'
    )
    parser.add_argument(
        '--end_dt', metavar='YYYYMMDDTHHMMSS', default=None,
        help='Ending time in ISO format'
    )
    parser.add_argument(
        '--t_deltat', metavar='HOURS', type=int, default=3,
        help='Time increment in hours (default: %(default)s)'
    )
    parser.add_argument(
        '-o', '--oname', metavar='ONAME', default='%Y%m%dT%H%M%S.png',
        help='Output filename (default: %(default)s)'
    )
    parser.add_argument(
        '-m', '--match', metavar='MATCH', default='%d:01,06,11,16,21,26',
        help='Output filename (default: %(default)s)'
    )
    parser.add_argument(
        '-b', '--basemap', metavar='BASEMAP', default=None,
        help='Basemap name (default: %(default)s)'
    )
    parser.add_argument(
        '--no_label', action='store_true', help='Label on/off flag'
    )
    parser.add_argument(
        '--no_logo', action='store_true', help='Logo on/off flag'
    )
    parser.add_argument(
        '--tight', action='store_true', help='Tight colorbar'
    )
    parser.add_argument(
        '--no_title', action='store_true', help='Title on/off flag'
    )
    parser.add_argument(
        '--plot_only', action='store_true', help='Plot only flag'
    )
    parser.add_argument(
        '--lights_off', action='store_true', help='Lights on/off flag'
    )
    parser.add_argument(
        '--label_size', metavar='LABEL_SIZE', default=None,
        help='Size of lat/lon labels (default: %(default)s)'
    )
    parser.add_argument(
        '--tick_label_size', metavar='TICK_LABEL_SIZE', default=None,
        help='Size of tick labels (default: %(default)s)'
    )
    parser.add_argument(
        '--track', metavar='PATHNAME', default=[], action='append',
        help='pathname or expression'
    )
    parser.add_argument(
        '--navigate', metavar='NAVIGATE', default='on',
        help='navigate on/off (default: %(default)s)'
    )
    parser.add_argument(
        '--fullframe', action='store_true', help='Turn full frame on'
    )

    return parser


def parse_args(args: list[str] | None = None) -> dict:

    if args is None:
        args = sys.argv[1:]

    parser = build_parser()

    if not args:
        parser.print_help()
        sys.exit(1)

    p_args = vars(parser.parse_args(args))

    if p_args['time_dt'] is None:
        now = dt.datetime.utcnow()
        hour = (now.hour // 12) * 12
        p_args['time_dt'] = now.strftime(f'%Y%m%dT{hour * 10000:06d}')

    p_args['time_dt'] = make_dt(p_args['time_dt'])
    p_args['start_dt'] = make_dt(p_args['start_dt'] or p_args['time_dt'])
    p_args['end_dt'] = make_dt(p_args['end_dt'] or p_args['start_dt'])

    if p_args['fcst_dt']:
        p_args['fcst_dt'] = make_dt(p_args['fcst_dt'])
        p_args['fcst_dt'] = dt.datetime.strptime(p_args['fcst_dt'], '%Y%m%dT%H%M%S')

    p_args['time_dt'] = dt.datetime.strptime(p_args['time_dt'], '%Y%m%dT%H%M%S')
    p_args['start_dt'] = dt.datetime.strptime(p_args['start_dt'], '%Y%m%dT%H%M%S')
    p_args['end_dt'] = dt.datetime.strptime(p_args['end_dt'], '%Y%m%dT%H%M%S')
    p_args['t_deltat'] = dt.timedelta(hours=p_args['t_deltat'])

    if p_args.get('fullframe', False):
        p_args['parea'] = '0 11 0 8.5'

    p_args['user'] = _record_user_activity(args, p_args)

    return p_args


def _record_user_activity(args: list[str], p_args: dict) -> dict:

    s_args = {v.lstrip('-') for v in args if v.startswith('-')}

    tracked_flags = {
        'region': {'region', 'r'},
        'field': {'field', 'f'},
        'plot': {'plot', 'p'},
        'level': {'level', 'l'},
        'stream': {'stream', 's'},
        'oname': {'oname', 'o'},
        'collection': {'collection', 'c'},
        'time_dt': {'time_dt'},
        'start_dt': {'start_dt'},
        'end_dt': {'end_dt'},
        't_deltat': {'t_deltat'},
    }

    user = {key: 1 for key, flags in tracked_flags.items() if flags & s_args}

    if user.get('plot'):
        user['field'] = 1
        p_args['field'] = p_args['plot']

    return user


def make_dt(dt_string: str) -> str:

    if len(dt_string) == 15:
        return dt_string

    if len(dt_string) <= 8:
        dt_string += 'T000000'
    else:
        dt_string += '000000'

    return dt_string[:14]

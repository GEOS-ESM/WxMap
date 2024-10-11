import importlib
import six
import re
import os
import sys
import copy
import yaml
import json
from string import Template

import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger()
logger.setLevel(logging.INFO)

fileFormatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log_file="/var/log/fluid/devlog/wxmap_dev.log"

directory = os.path.dirname(__file__)
here = os.path.abspath(directory)

#log_file=f'{here}/../../logs/wxmap_dev.log'
#try: 
#    fileHandler = RotatingFileHandler(log_file, maxBytes=1024, backupCount=10)
#except Exception:
#    log_file = f'{here}/../../logs/wxmap_dev.log'
#    fileHandler = RotatingFileHandler(log_file, maxBytes=1024, backupCount=10)
#
#fileHandler.setFormatter(fileFormatter)
#logger.addHandler(fileHandler)

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

novalue    = object()
config_cache = {}
front_end = '/explore/dataportal/applications/GMAO/fluid{server}/fluid_{server}/data_services/'
current_dir = os.getcwd()
if 'fluidprod' in current_dir:
    server='prod'
else:
    server='dev'
replace_dict={
        '/portal/web/cgi-bin/gmao/data-services/data_services/':front_end.format(server=server),
        '/dataportal01/devel/gmao_data_services/config/wxmaps/share/':'/explore/dataportal/applications/devel/gmao_data_services/config/wxmaps/share/',
        '/dataportal01/devel/gmao_data_services/static/data-services':'/explore/dataportal/applications/devel/gmao_data_services/static/data-services',
        }
class Config(dict):
    """
    Attributes
    ---------- 
    registry: dict
    
    
    Methods
    ----------
    __call__ method:
        get_config(pathname,default=novalue)
    find(path, name, depth=-1, cfg=None)
    follow(paths)
    get_values(pathname,default=None)
    get_keys(pathname,default=None)
    get_items(pathname,default=None,flat=True,hide=True)
    expand(paths)
    mkpath(root,path)
    read(file)
    readJSON(file)
    read_resolve(file,**kwargs)
    replace(s,**defs)
    mount(cfg,root=None)
    ispartition(dir)
    fcopy(path)
    fdcopy(path)
    serialize(hash)
    deserialize(hash)
    copy_yaml(hash)
    checkNode(config)
    
    
    """

    def __init__(self, *args, **kw):

        self.registry = {}
        config_cache = {}
        super(Config,self).__init__(*args, **kw)

#------------------------------------------------------------------------------

    def find(self, path, name, depth=-1, cfg=None):

        result = []

        if cfg is None:
            cfg = self.follow(path)

        for key in cfg:

            apath = path + [key]

            if key == name:
                result.append(apath)

            if depth == 0:
                continue

            if self.ispartition(cfg[key]):
                result += self.find(apath,name,depth,cfg=cfg[key])
            elif isinstance(cfg[key], dict):
                result += self.find(apath,name,depth-1,cfg=cfg[key])

        return result

#------------------------------------------------------------------------------

    def follow(self, paths):

        cfg   = self
        apath = ''

        for path in paths:

            if path == '/':
                continue

            apath += '/' + str(path)

            if path not in cfg:
                raise KeyError('Config.follow: "' + apath +
                               '" No such file or directory')

            if isinstance(cfg[path],dict):
                cfg = cfg[path]
            else:
                raise KeyError('Config.follow: "' + apath +
                                          '" is not a directory')

        return cfg
    
#------------------------------------------------------------------------------

    def get_values(self, pathname, default=None):

        return self.get_items(pathname, default).values()

#------------------------------------------------------------------------------
    
    def get_keys(self, pathname, default=None):

        return self.get_items(pathname, default).keys()

#------------------------------------------------------------------------------

    def get_items(self, pathname, default=None, flat=True, hide=True):

        items     = {}
        pathnames = pathname

        if not isinstance(pathname,list):
            return items

        if not isinstance(pathname[0],list):
            pathnames = [pathname]
        
        for pn in pathnames:

            path = pn[0:-1]
            name = pn[-1]
            key  = pn[-2]
            cfg  = self.follow(path)

            if hide and cfg.get('hide','no') == 'yes': continue

            hash = items
            if not flat: hash = self.mkpath(hash,path[1:-1])

            if name not in cfg:
                hash[key] = default
            else:
                hash[key] = cfg[name]

        return items

#------------------------------------------------------------------------------    
    
    def get_config(self, pathname, default=novalue):

        pathname = self.expand(pathname)

        try:
            cfg = self.follow(pathname[0:-1])
        except KeyError:
            if default is novalue:
                raise
            else:
                return default

        if default is novalue:
            return cfg[pathname[-1]]
        else:
            return cfg.get(pathname[-1],default)

    __call__ = get_config

#------------------------------------------------------------------------------    
    
    def expand(self, paths):

        pathname = []
        if not isinstance(paths,list):
            paths = [paths]

        for path in paths:

            if isinstance(path,six.string_types):
                pathname += path.split('/')
            else:
                pathname.append(path)

        return pathname

#------------------------------------------------------------------------------

    def mkpath(self, root, path):

        for dir in path:

            if dir not in root:
                root[dir] = {}

            root = root[dir]

        return root

#------------------------------------------------------------------------------    
    
    def read(self, file, **add):
    
        if file in config_cache: 
            config_cache[file].update(add)
            return config_cache[file]
    
        yaml.warnings({'YAMLLoadWarning': False})
        with open(file, 'r') as ymlfile:
            config = self.copy_yaml(yaml.load(ymlfile,Loader))

        config.update(add)
        config_cache[file]=  self.checkNode(config)
        config_cache[file]= config
        
        return config_cache[file]

#------------------------------------------------------------------------------

    def readJSON(self, file):

        #if file in config_cache: return config_cache[file]

        with open(file, 'r') as jsonfile:
            config             = json.load(jsonfile)
            
        config = self.deserialize(config)
        config_cache[file]=self.checkNode(config)
        config_cache[file] = config
        return config_cache[file]
    
#------------------------------------------------------------------------------
    
    def read_resolve(self, file, **kwargs):
        """Reads a YAML file and resolves/interpolates all defined variables
        in the configuration using environment settings and root-level
        parameters values.

        Returns
        -------
        d: dict
          YAML dictionary with all defined variables interpolated.
        """

        # Read input file as YAML file
        with open(file) as f:
            input_defs = yaml.load(f,Loader=yaml.UnsafeLoader)

        # Extract definitions
        defs = { k:str(v) for k,v in six.iteritems(os.environ) }
        defs.update(kwargs)
        defs.update( {k:str(v) for k,v in six.iteritems(input_defs)
          if not isinstance(v,dict) and not isinstance(v,list)} )

        # Read input file as text file
        with open(file) as f:
            text = f.read()

        # Replace any unresolved variables in the file
        text = self.replace(text, **defs)

        # Return a yaml
        return yaml.load(text,Loader=yaml.UnsafeLoader)

    def replace(self, s, **defs):
        
        """Interpolate/replace variables in string

        Resolved variable formats are: $var, {{var}} and $(var). Undefined
        variables remain unchanged in the returned string. This method will
        recursively resolve variables of variables.

        Parameters
        ----------
        s : string, required
          Input string containing variables to be resolved.
        defs: dict, required
          dictionary of definitions for resolving variables expressed
          as key-word arguments.

        Returns
        -------
        s_interp: string
          Interpolated string. Undefined variables are left unchanged.
        """

        expr = s

        # Resolve special variables: {{var}}
        for var in re.findall(r'{{(\w+)}}', expr):
            if var in defs:
                expr = re.sub(r'{{'+var+'}}',defs[var],expr)

        # Resolve special variables: $(var)
        for var in re.findall(r'\$\((\w+)\)', expr):
            if var in defs:
                expr = re.sub(r'\$\('+var+r'\)',defs[var],expr)

        # Resolve defs
        s_interp = Template(expr).safe_substitute(defs)

        # Recurse until no substitutions remain
        if s_interp != s:
            s_interp = self.replace(s_interp, **defs)

        return s_interp
    
#------------------------------------------------------------------------------

    def mount(self, cfg, root=None):

        hash = self

        if root is not None:

            for dir in root.split('/'):

                if dir == '/': continue
                if not dir: continue

                if dir not in hash:
                    hash[dir] = {}
                elif not isinstance(hash[dir], dict):
                    hash[dir] = {}

                hash = hash[dir]

        self.overlay(hash,cfg)
    
#------------------------------------------------------------------------------

    def ispartition(self, dir):

        if not isinstance(dir,dict):
            return False

        result = [key for key in dir.keys() if not isinstance(dir[key],dict)]

        if result:
            return False

        return True
    
#------------------------------------------------------------------------------

    def fcopy(self, path):

        flat_list = {}
        hash      = self

        for dir in path:
            hash = hash.get(dir,{})
            flat_list.update(hash)
            if dir in flat_list:
                del flat_list[dir]

        return flat_list
    
#------------------------------------------------------------------------------

    def fdcopy(self, path):
        return copy.deepcopy(self.fcopy(path))
    
#------------------------------------------------------------------------------

    def overlay(self, hash1, hash2):

        for key2 in hash2:

            if key2 not in hash1:
                if isinstance(hash2[key2], dict):
                    hash1[key2] = copy.deepcopy(hash2[key2])
                else:
                    hash1[key2] = hash2[key2]
            elif isinstance(hash2[key2],dict) and isinstance(hash1[key2],dict):
                self.overlay(hash1[key2], hash2[key2])
            else:
                hash1[key2] = hash2[key2]
    
#------------------------------------------------------------------------------

    def serialize(self, hash):

        for key in hash:

            if isinstance(hash[key],dict):
                self.serialize(hash[key])

            try:
                json.dumps(hash[key])
            except TypeError:
                hash[key] = hash[key].__module__ + '.' + \
                            hash[key].__class__.__name__ + \
                            ' []'
    
#------------------------------------------------------------------------------

    def deserialize(self, hash):

        for key in hash:

            if isinstance(hash[key],dict):
                self.deserialize(hash[key])

            if self.is_object_string(hash[key]):
                object      = hash[key]
                module_name = object.split('.')[0]
                class_name  = object.split('.')[1].split()[0]

                try:
                    module      = importlib.import_module(module_name)
                    class_      = getattr(module, class_name)
                    hash[key]   = class_()
                except Exception as e:
                    hash[key]   = object

        return hash
    
#------------------------------------------------------------------------------

    def is_object_string(self, value):

        if not isinstance(value, six.string_types): return False
        match = re.match(r'\w+\.\w+ \[\]', value)
        if match: return True

        return False
    
#------------------------------------------------------------------------------

    def copy_yaml(self, hash):

        hashkeys=list(hash.keys())
        for key in hashkeys:

            new_key = str(key)

            if not isinstance(key, six.string_types):
                hash[new_key] = hash[key]
                del hash[key]

            if isinstance(hash[new_key], dict):

                if id(hash[new_key]) in self.registry:
                    hash[new_key] = dict(hash[new_key])
                    self.registry[id(hash[new_key])] = 1
                else:
                    self.registry[id(hash[new_key])] = 1

                self.copy_yaml(hash[new_key])

        return hash
    
#------------------------------------------------------------------------------
    
    def checkNode(self,config):
        
        def replace_dportal(path):
            for k,v in replace_dict.items():
                if k in path:
                    path=path.replace(k,v)
            return path
        
        import platform
        node=platform.node()
        
        #if 'dphttp' in node:
            #return

        if 'stream' in config:
            for k,v in config['stream'].items():
                if 'uri' in v:
                    config['stream'][k]['uri']=replace_dportal(v['uri'])
        
        for k,v in config.items():
            if 'font' in k or 'path' in k:
                config[k]=replace_dportal(v)
                
        return config
    
#------------------------------------------------------------------------------

    #def __deepcopy__(self, memo):
    #    cls = self.__class__
    #    result = cls.__new__(cls)
    #    memo[id(self)] = result
    #    for k, v in self.__dict__.items():
    #        setattr(result, k, copy.deepcopy(v, memo))
    #    return result
    
    
class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class UsageError(Error):
    """Exception raise for errors in the input."""
    def __init__(self, msg):
        self.msg = msg

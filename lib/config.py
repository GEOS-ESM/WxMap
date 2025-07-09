import importlib
import os
import re
import sys
import copy
import yaml
import json
import platform
import datetime as dt
from string import Template
from typing import Any, Dict, List, Optional, Union

from logging_config import logger

# YAML Loader and Dumper
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

novalue = object()
config_cache: Dict[str, Any] = {}

class Config(dict):
    """
    A configuration class for handling YAML and JSON configurations with variable interpolation and merging capabilities.

    Attributes:
        registry (dict): Registry to keep track of objects during copy operations.

    Methods:
        find(path, name, depth=-1, cfg=None): Find all paths to a given key.
        follow(paths): Follow a path and return the configuration at that path.
        get_values(pathname, default=None): Get values from the configuration.
        get_keys(pathname, default=None): Get keys from the configuration.
        get_items(pathname, default=None, flat=True, hide=True): Get items from the configuration.
        expand(paths): Expand paths into a list of path components.
        mkpath(root, path): Create a path in the configuration.
        read(file, **add): Read a YAML configuration file.
        readJSON(file): Read a JSON configuration file.
        read_resolve(file, **kwargs): Read and resolve variables in a YAML file.
        replace(s, **defs): Replace variables in a string.
        mount(cfg, root=None): Mount a configuration at a given root.
        is_partition(dir): Check if a directory is a partition.
        fcopy(path): Make a flat copy of the configuration at a path.
        fdcopy(path): Make a deep copy of the configuration at a path.
        overlay(hash1, hash2): Overlay one configuration onto another.
        serialize(hash): Serialize the configuration for JSON dumping.
        deserialize(hash): Deserialize a configuration from JSON.
        copy_yaml(hash): Make a copy of a YAML configuration.
        check_node(config): Check and modify paths in the configuration based on the node.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.registry: Dict[int, int] = {}
        super().__init__(*args, **kwargs)
        self.get_node()

    def get_node(self) -> None:
        """
        Get the node name.

        Returns:
            The node name.
        """
        node = platform.node()
        if 'dplogin' in node:
            self.node = 'dataportal'
        elif 'discover' in node:
            self.node = 'discover'
        else:
            self.node = 'local'

    def find(self, path: List[str], name: str, depth: int = -1, cfg: Optional[Dict] = None) -> List[List[str]]:
        """
        Find all paths to a given key in the configuration.

        Args:
            path: The path to start searching from.
            name: The name of the key to find.
            depth: The depth to search (-1 for unlimited).
            cfg: The configuration dictionary to search in.

        Returns:
            A list of paths where the key is found.
        """
        result = []

        if cfg is None:
            cfg = self.follow(path)

        for key in cfg:
            apath = path + [key]

            if key == name:
                result.append(apath)

            if depth == 0:
                continue

            if self.is_partition(cfg[key]):
                result += self.find(apath, name, depth, cfg=cfg[key])
            elif isinstance(cfg[key], dict):
                result += self.find(apath, name, depth - 1, cfg=cfg[key])

        return result

    def follow(self, paths: Union[str, List[str]]) -> Dict:
        """
        Follow a path in the configuration and return the configuration at that path.

        Args:
            paths: The path to follow.

        Returns:
            The configuration dictionary at the specified path.

        Raises:
            KeyError: If the path does not exist.
        """
        cfg = self
        apath = ''

        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            if path == '/':
                continue

            apath += '/' + str(path)

            if path not in cfg:
                raise KeyError(f'Config.follow: "{apath}" No such file or directory')

            if isinstance(cfg[path], dict):
                cfg = cfg[path]
            else:
                raise KeyError(f'Config.follow: "{apath}" is not a directory')

        return cfg

    def get_values(self, pathname: List[List[str]], default: Any = None) -> List[Any]:
        """
        Get values from the configuration based on the given paths.

        Args:
            pathname: List of paths.
            default: Default value if key is not found.

        Returns:
            List of values.
        """
        return list(self.get_items(pathname, default).values())

    def get_keys(self, pathname: List[List[str]], default: Any = None) -> List[str]:
        """
        Get keys from the configuration based on the given paths.

        Args:
            pathname: List of paths.
            default: Default value if key is not found.

        Returns:
            List of keys.
        """
        return list(self.get_items(pathname, default).keys())

    def get_items(
        self,
        pathname: Union[List[str], List[List[str]]],
        default: Any = None,
        flat: bool = True,
        hide: bool = True
    ) -> Dict:
        """
        Get items from the configuration based on the given paths.

        Args:
            pathname: List of paths.
            default: Default value if key is not found.
            flat: Whether to return a flat dictionary.
            hide: Whether to skip hidden items.

        Returns:
            Dictionary of items.
        """
        items = {}
        pathnames = pathname

        if not isinstance(pathname, list):
            return items

        if not isinstance(pathname[0], list):
            pathnames = [pathname]

        for pn in pathnames:
            path = pn[:-1]
            name = pn[-1]
            key = pn[-2]
            cfg = self.follow(path)

            if hide and cfg.get('hide', 'no') == 'yes':
                continue

            hash_map = items
            if not flat:
                hash_map = self.mkpath(hash_map, path[1:-1])

            if name not in cfg:
                hash_map[key] = default
            else:
                hash_map[key] = cfg[name]

        return items

    def get_config(self, pathname: Union[str, List[str]], default: Any = novalue) -> Any:
        """
        Get a configuration value based on the given path.

        Args:
            pathname: The path to the configuration value.
            default: Default value if key is not found.

        Returns:
            The configuration value.
        """
        pathname = self.expand(pathname)

        try:
            cfg = self.follow(pathname[:-1])
        except KeyError:
            if default is novalue:
                raise
            else:
                return default

        if default is novalue:
            return cfg[pathname[-1]]
        else:
            return cfg.get(pathname[-1], default)

    __call__ = get_config

    def expand(self, paths: Union[str, List[Union[str, int]]]) -> List[Union[str, int]]:
        """
        Expand a path or list of paths into a list of path components.

        Args:
            paths: The path(s) to expand.

        Returns:
            A list of path components.
        """
        pathname = []
        if not isinstance(paths, list):
            paths = [paths]

        for path in paths:
            if isinstance(path, str):
                pathname += path.split('/')
            else:
                pathname.append(path)

        return pathname

    def mkpath(self, root: Dict, path: List[str]) -> Dict:
        """
        Create a path in the configuration dictionary.

        Args:
            root: The root dictionary.
            path: The path components.

        Returns:
            The dictionary at the end of the path.
        """
        for dir_name in path:
            if dir_name not in root:
                root[dir_name] = {}
            root = root[dir_name]
        return root

    def read(self, file: str, **add) -> Dict:
        """
        Read a YAML configuration file.

        Args:
            file: The file path.
            **add: Additional key-value pairs to add to the configuration.

        Returns:
            The configuration dictionary.
        """
        if file in config_cache:
            config_cache[file].update(add)
            return config_cache[file]

        with open(file, 'r') as ymlfile:
            yaml_content = yaml.load(ymlfile, Loader=Loader)
            config_data = self.copy_yaml(yaml_content)

        config_data.update(add)
        config_cache[file] = self.check_node(config_data)
        return config_cache[file]

    def readJSON(self, file: str) -> Dict:
        """
        Read a JSON configuration file.

        Args:
            file: The file path.

        Returns:
            The configuration dictionary.
        """
        with open(file, 'r') as jsonfile:
            config_data = json.load(jsonfile)

        config_data = self.deserialize(config_data)
        config_cache[file] = self.check_node(config_data)
        return config_cache[file]

    def read_resolve(self, file: str, **kwargs) -> Dict:
        """
        Read a YAML file and resolve/interpolate all defined variables in the configuration.

        Args:
            file: The file path.
            **kwargs: Additional definitions for variable substitution.

        Returns:
            The configuration dictionary with variables resolved.
        """
        # Read input file as YAML
        with open(file) as f:
            input_defs = yaml.load(f, Loader=Loader)

        # Extract definitions
        defs = {k: str(v) for k, v in os.environ.items()}
        defs.update(kwargs)
        defs.update(
            {k: str(v) for k, v in input_defs.items() if not isinstance(v, (dict, list))}
        )

        # Read input file as text
        with open(file) as f:
            text = f.read()

        # Replace any unresolved variables in the file
        text = self.replace(text, **defs)

        # Return the resolved YAML content
        return yaml.load(text, Loader=Loader)

    def replace(self, s: str, **defs) -> str:
        """
        Interpolate/replace variables in a string.

        Resolved variable formats are: $var, {{var}}, and $(var).
        Undefined variables remain unchanged.

        Args:
            s: The input string containing variables to be resolved.
            **defs: Dictionary of definitions for resolving variables.

        Returns:
            The interpolated string.
        """
        expr = s

        # Resolve special variables: {{var}}
        for var in re.findall(r'{{(\w+)}}', expr):
            if var in defs:
                expr = re.sub(r'{{' + var + '}}', defs[var], expr)

        # Resolve special variables: $(var)
        for var in re.findall(r'\$\((\w+)\)', expr):
            if var in defs:
                expr = re.sub(r'\$\(' + var + r'\)', defs[var], expr)

        # Resolve defs
        s_interp = Template(expr).safe_substitute(defs)

        # Recurse until no substitutions remain
        if s_interp != s:
            s_interp = self.replace(s_interp, **defs)

        return s_interp

    def mount(self, cfg: Dict, root: Optional[str] = None) -> None:
        """
        Mount a configuration at a given root.

        Args:
            cfg: The configuration dictionary to mount.
            root: The root path where the configuration should be mounted.
        """
        hash_map = self

        if root is not None:
            for dir_name in root.split('/'):
                if dir_name == '/':
                    continue
                if not dir_name:
                    continue

                if dir_name not in hash_map:
                    hash_map[dir_name] = {}
                elif not isinstance(hash_map[dir_name], dict):
                    hash_map[dir_name] = {}

                hash_map = hash_map[dir_name]

        self.overlay(hash_map, cfg)

    def is_partition(self, dir_entry: Any) -> bool:
        """
        Check if a directory is a partition (contains only directories).

        Args:
            dir_entry: The directory entry to check.

        Returns:
            True if the directory is a partition, False otherwise.
        """
        if not isinstance(dir_entry, dict):
            return False

        result = [key for key in dir_entry.keys() if not isinstance(dir_entry[key], dict)]

        return not bool(result)

    def fcopy(self, path: List[str]) -> Dict:
        """
        Make a flat copy of the configuration at the specified path.

        Args:
            path: The path to copy.

        Returns:
            A flat dictionary of the configuration.
        """
        flat_list = {}
        hash_map = self

        for dir_name in path:
            hash_map = hash_map.get(dir_name, {})
            flat_list.update(hash_map)
            if dir_name in flat_list:
                del flat_list[dir_name]

        return flat_list

    def fdcopy(self, path: List[str]) -> Dict:
        """
        Make a deep copy of the configuration at the specified path.

        Args:
            path: The path to copy.

        Returns:
            A deep copy of the configuration.
        """
        return copy.deepcopy(self.fcopy(path))

    def overlay(self, hash1: Dict, hash2: Dict) -> None:
        """
        Overlay one configuration onto another.

        Args:
            hash1: The base configuration dictionary.
            hash2: The configuration dictionary to overlay.
        """
        for key2 in hash2:
            if key2 not in hash1:
                if isinstance(hash2[key2], dict):
                    hash1[key2] = copy.deepcopy(hash2[key2])
                else:
                    hash1[key2] = hash2[key2]
            elif isinstance(hash2[key2], dict) and isinstance(hash1[key2], dict):
                self.overlay(hash1[key2], hash2[key2])
            else:
                hash1[key2] = hash2[key2]

    def serialize(self, hash_map: Dict) -> None:
        """
        Serialize the configuration for JSON dumping.

        Args:
            hash_map: The configuration dictionary to serialize.
        """
        for key in hash_map:
            if isinstance(hash_map[key], dict):
                self.serialize(hash_map[key])

            try:
                json.dumps(hash_map[key])
            except TypeError:
                hash_map[key] = (
                    f"{hash_map[key].__module__}."
                    f"{hash_map[key].__class__.__name__} []"
                )

    def deserialize(self, hash_map: Dict) -> Dict:
        """
        Deserialize a configuration from JSON.

        Args:
            hash_map: The configuration dictionary to deserialize.

        Returns:
            The deserialized configuration dictionary.
        """
        for key in hash_map:
            if isinstance(hash_map[key], dict):
                self.deserialize(hash_map[key])

            if self.is_object_string(hash_map[key]):
                obj_str = hash_map[key]
                module_name, class_name = obj_str.split('.')[0], obj_str.split('.')[1].split()[0]

                try:
                    module = importlib.import_module(module_name)
                    class_ = getattr(module, class_name)
                    hash_map[key] = class_()
                except Exception as e:
                    logger.error(f"Error deserializing object '{obj_str}': {e}")
                    hash_map[key] = obj_str

        return hash_map

    def is_object_string(self, value: Any) -> bool:
        """
        Check if a value is a string representation of an object.

        Args:
            value: The value to check.

        Returns:
            True if the value is an object string, False otherwise.
        """
        if not isinstance(value, str):
            return False
        match = re.match(r'\w+\.\w+ \[\]', value)
        return bool(match)

    def copy_yaml(self, hash_map: Dict) -> Dict:
        """
        Make a copy of a YAML configuration.

        Args:
            hash_map: The configuration dictionary to copy.

        Returns:
            The copied configuration dictionary.
        """
        hash_keys = list(hash_map.keys())
        for key in hash_keys:
            new_key = str(key)

            if not isinstance(key, str):
                hash_map[new_key] = hash_map[key]
                del hash_map[key]

            if isinstance(hash_map[new_key], dict):
                if id(hash_map[new_key]) in self.registry:
                    hash_map[new_key] = dict(hash_map[new_key])
                    self.registry[id(hash_map[new_key])] = 1
                else:
                    self.registry[id(hash_map[new_key])] = 1

                self.copy_yaml(hash_map[new_key])

        return hash_map

    def check_node(self, config: Dict) -> Dict:
        """
        Check and modify paths in the configuration based on the node.

        Args:
            config: The configuration dictionary.

        Returns:
            The modified configuration dictionary with uri key
            that corresponds to the current server.
        """
        if self.node == 'dataportal':
            uri = 'uri_dataportal'
        elif self.node == 'discover':
            uri = 'uri_discover'
        else:
            uri = 'uri'

        if 'stream' in config:
            for k, v in config['stream'].items():
                if 'uri' in v:
                    config['stream'][k]['uri'] = v[uri]

        return config


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class UsageError(Error):
    """Exception raised for errors in the input."""

    def __init__(self, msg: str) -> None:
        self.msg = msg

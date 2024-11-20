import os
import sys
from glob import glob
from typing import Dict, Any, List

# Attempt to use ruamel.yaml, fall back to pyyaml if unavailable
try:
    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.preserve_quotes = True  # Preserve quotes in the original YAML
except ImportError:
    import yaml

    class YamlWrapper:
        """
        A wrapper class to mimic ruamel.yaml behavior with PyYAML.
        """
        @staticmethod
        def load(stream):
            return yaml.safe_load(stream)

        @staticmethod
        def dump(data, stream):
            yaml.dump(data, stream, default_flow_style=False, sort_keys=False)

    yaml = YamlWrapper()

# Base directory and subpaths to search
base_path = "/explore/dataportal/applications/GMAO/fluiddev/config/config_dev/share"            
subpaths = ['wxmaps', 'themes', 'missions', 'examples']  

def update_paths(entry: Dict[str, Any]) -> None:
    """
    Update the `uri_discover` and `uri_dataportal` paths in a stream entry.
    
    Args:
        entry (Dict[str, Any]): The dictionary representing a single stream entry in the YAML file.
    """
    if 'uri' in entry:
        if 'uri_discover2' not in entry:
            entry['uri_discover'] = entry['uri'].replace("/gmao/merra2/pub/", "/gmao/merra2/data/pub/")
            if entry['uri_discover'].endswith(".portal"):
                entry['uri_discover'] = entry['uri_discover'].replace(".portal", "")
        
        if 'uri_dataportal2' not in entry:
            entry['uri_dataportal'] = entry['uri'].replace("/gmao/merra2/data/pub/", "/gmao/merra2/pub/")

def process_stream_file(filepath: str) -> None:
    """
    Process a single `stream.yml` file to update paths for each stream entry.
    
    Args:
        filepath (str): The path to the `stream.yml` file.
    """
    with open(filepath, "r") as file:
        data = yaml.load(file)

    for stream_name, stream_info in data.get('stream', {}).items():
        update_paths(stream_info)

    with open(filepath, "w") as file:
        yaml.dump(data, file)

    print(f"Updated: {filepath}")    

def find_and_process_stream_files(base_path: str, subpaths: List[str]) -> None:
    """
    Recursively search for all `stream.yml` files within a list of directories and process each.

    Args:
        base_path (str): The base directory path.
        subpaths (List[str]): A list of subdirectory paths to search for `stream.yml` files.
    """
    for subpath in subpaths:
        directory = os.path.join(base_path, subpath)
        for filepath in glob(os.path.join(directory, "**", "stream.yml"), recursive=True):
            process_stream_file(filepath)

# Run the script
if __name__ == '__main__':
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
        if len(sys.argv) > 2:
            subpaths = sys.argv[2:]
            
    find_and_process_stream_files(base_path, subpaths)

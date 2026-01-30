"""
Tyaml tracks changes in a YAML file by storing its unique states in a companion history file.
Each state is identified by its hash and timestamp, with optional indexing for versioning.

Example:
    ty = Tyaml('config.yaml')
    ty.track(1)  # Save version if content changed, with index = 1
"""

import hashlib
import yaml
import time
from pathlib import Path


class Tyaml:
    def __init__(self, yaml_path, history_path=None):

        if not isinstance(yaml_path, (str, Path)):
            raise TypeError("yaml_path must be a string or Path object")
        self.yaml_path = Path(yaml_path).resolve()
        if not self.yaml_path.is_file():
            raise FileNotFoundError(f"{self.yaml_path} is not a file.")

        self.hash
            
        # Create history path by adding -hist before the extension
        stem = self.yaml_path.stem + '-hist'
        if history_path is not None:
            self.history_path = Path(history_path).resolve()
        else:
            self.history_path = self.yaml_path.with_name(stem + self.yaml_path.suffix)

    @property
    def hash(self):
        # Load and validate YAML content, and compute file hash

        with open(self.yaml_path, 'rb') as f:
            content = f.read()
            try:
                self.data = yaml.safe_load(content.decode('utf-8'))
            except yaml.YAMLError as e:
                raise ValueError(f"{self.yaml_path} is not a valid YAML file: {e}")
            return hashlib.sha256(content).hexdigest()

    def track(self, index=None):
        """
        Track changes to the YAML file. Appends a new entry to the history if the hash is new.
        If index is provided, it must be an integer strictly greater than the previous one.
        """
        hash = self.hash  # Recompute hash to check for changes
    
        if self.history_path.exists():
            with open(self.history_path, 'r') as f:
                try:
                    history = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    raise ValueError(f"{self.history_path} is not a valid YAML file: {e}")
                if not isinstance(history, list):
                    raise ValueError(f"{self.history_path} must contain a YAML list.")
        else:
            history = []

        # If previous entries used indexing, require index for consistency
        if history and 'index' in history[-1] and index is None:
            raise ValueError("Index is required because previous history entries include it.")

        last_hash = history[-1]['hash'] if history else None
        last_index = history[-1]['index'] if history and 'index' in history[-1] else -1

        if index is not None:
            assert isinstance(index, int), "index must be an integer"
            assert index > last_index, "index must be greater than the last recorded index"

        if hash != last_hash:
            entry = {
                'hash': hash,
                'data': self.data,
                'timestamp': int(time.time()),
                'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
            }
            if index is not None:
                entry['index'] = index

            history.append(entry)

            with open(self.history_path, 'w') as f:
                yaml.safe_dump(history, f)

        return hash
    
    def clear(self):
        """
        Clear the history file.
        """
        if self.history_path.exists():
            self.history_path.unlink()

            
        

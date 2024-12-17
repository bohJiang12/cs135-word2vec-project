"""
Utility module for helping
- computing cosine similarity
"""
from typing import Union
from pathlib import Path

import yaml


class Config:
    """Model training config class"""
    def __init__(self, config_file: Union[str, Path]):
        with open(config_file, 'r') as f:
            self.params = yaml.safe_load(f)
    
    def __str__(self) -> str:
        config_info = f"Config:\n{'=' * 50}"
        for k, v in self.params.items():
            config_info += f"{k}: {v}\n"
        return config_info
        






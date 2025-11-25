"""
Configuration loader for managing framework configurations.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Configuration loader that supports YAML and JSON formats.
    
    Provides methods to load, merge, and access configuration data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the config loader.
        
        Args:
            config: Optional initial configuration dictionary
        """
        self._config: Dict[str, Any] = config or {}

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "ConfigLoader":
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file (YAML or JSON)
            
        Returns:
            ConfigLoader instance with loaded configuration
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.suffix in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            elif file_path.suffix == ".json":
                config = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_path.suffix}. "
                    "Supported formats: .yaml, .yml, .json"
                )
        
        logger.info(f"Loaded configuration from {file_path}")
        return cls(config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ConfigLoader":
        """
        Create a ConfigLoader from a dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            ConfigLoader instance
        """
        return cls(config)

    def merge(self, other: Union["ConfigLoader", Dict[str, Any]]) -> "ConfigLoader":
        """
        Merge another configuration into this one.
        
        Args:
            other: Another ConfigLoader or dictionary to merge
            
        Returns:
            New ConfigLoader with merged configuration
        """
        if isinstance(other, ConfigLoader):
            other_config = other._config
        else:
            other_config = other
        
        merged = self._deep_merge(self._config.copy(), other_config)
        return ConfigLoader(merged)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., "model.name" or "model")
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., "model.name")
            value: Value to set
        """
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Get the configuration as a dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    def save(self, file_path: Union[str, Path], format: str = "yaml") -> None:
        """
        Save configuration to a file.
        
        Args:
            file_path: Path to save the configuration file
            format: File format ("yaml" or "json")
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            if format.lower() in ["yaml", "yml"]:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
            elif format.lower() == "json":
                json.dump(self._config, f, indent=2, sort_keys=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved configuration to {file_path}")

    @staticmethod
    def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            update: Dictionary to merge into base
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Support dictionary-style assignment."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return self.get(key) is not None


def load_config(file_path: Union[str, Path]) -> ConfigLoader:
    """
    Convenience function to load configuration from a file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        ConfigLoader instance
    """
    return ConfigLoader.from_file(file_path)


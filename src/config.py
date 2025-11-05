"""Configuration management for bankruptcy prediction project."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from loguru import logger


class Config:
    """Configuration loader and manager."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_paths()
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Configuration loaded from {self.config_path}")
        return config
    
    def _setup_paths(self) -> None:
        """Create necessary directories."""
        paths = [
            self.config['data']['raw_data_path'],
            self.config['data']['processed_data_path'],
            self.config['data']['interim_data_path'],
            self.config['data']['external_data_path'],
            self.config['output']['models_path'],
            self.config['output']['figures_path'],
            self.config['output']['results_path'],
        ]
        
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)
            
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = self.config['output'].get('log_level', 'INFO')
        log_file = self.config['output'].get('log_file', 'bankruptcy_prediction.log')
        
        logger.add(
            log_file,
            rotation="10 MB",
            retention="10 days",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Dot-separated key path (e.g., 'data.raw_data_path')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Args:
            key: Dot-separated key path
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def save(self, path: str = None) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Optional path to save to (defaults to original path)
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        logger.info(f"Configuration saved to {save_path}")
    
    @property
    def random_seed(self) -> int:
        """Get random seed for reproducibility."""
        return self.config['project']['random_seed']
    
    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"
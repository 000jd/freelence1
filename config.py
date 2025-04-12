import yaml
import os


class Config:
    """Simple configuration class to load and manage parameters from YAML file"""

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize configuration from YAML file"""
        self.config_path = config_path
        self.config = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file) or {}

    def get(self, section, key=None, default=None):
        """Get configuration value"""
        # If section doesn't exist, return default
        if section not in self.config:
            return default

        section_data = self.config[section]

        # If no key specified, return the entire section
        if key is None:
            return section_data

        # If section is a dictionary and key exists, return the value
        if isinstance(section_data, dict) and key in section_data:
            return section_data[key]

        # Default case
        return default

    def update(self, section, key, value):
        """Update configuration value"""
        if section not in self.config:
            self.config[section] = {}

        self.config[section][key] = value

    def save(self, path=None):
        """Save configuration to YAML file"""
        save_path = path or self.config_path

        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)


def get_config(config_path='config.yaml'):
    """Get configuration instance"""
    return Config(config_path)

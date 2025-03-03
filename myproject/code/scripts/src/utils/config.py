import yaml
import os

def load_config(config_path):
    """
    Load a YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, config_path):
    """
    Save a configuration dictionary to a YAML file.
    """
    with open(config_path, "w") as f:
        yaml.dump(config, f)
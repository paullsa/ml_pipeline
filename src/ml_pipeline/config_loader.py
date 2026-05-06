import json


def load_config(config_path):
    """Load a JSON config file and return it as a dict.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, "r") as f:
        return json.load(f)

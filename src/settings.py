from pathlib import Path
import yaml


def load_settings() -> dict:
    """
    Loads the settings.yaml file from the project root and returns it as a dictionary.
    Raises exceptions if the file does not exist or is not valid YAML.
    """
    module_file = Path(__loader__.get_filename())
    root_folder = module_file.resolve().parent.parent
    settings_path = root_folder / 'settings.yaml'

    if not settings_path.is_file():
        raise FileNotFoundError(f"settings.yaml not found at {settings_path}")

    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"{settings_path} is not a valid YAML file: {e}")

    if not isinstance(settings, dict):
        raise ValueError(f"{settings_path} does not contain a valid YAML dictionary.")

    return settings
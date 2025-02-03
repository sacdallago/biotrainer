import shutil

from pathlib import Path
from typing import Any
from urllib import request
from urllib.parse import urlparse

from .config_exception import ConfigurationException


def download_file_from_config(option_name: str, url: str, config_file_path: Path):
    """
    Download the file from a URL if downloading is allowed.

    Args:
        option_name (str): The name of the option to download.
        url (str): The url to download the file from.
        config_file_path (Path): Path to the configuration file directory.
    Raises:
        Exception: If the file cannot be downloaded.
    """
    if is_url(url):
        try:
            print(f"Trying to download {option_name} from {url}")
            req = request.Request(url, headers={
                'User-Agent': "Mozilla/5.0 (Windows NT 6.1; Win64; x64)"
            })

            file_name = url.split("/")[-1]
            save_path = str(config_file_path / f"downloaded_{file_name}")
            with request.urlopen(req) as response, open(save_path, 'wb') as outfile:
                if response.status == 200:
                    print(f"OK - Downloading file {option_name} (size: {response.length / (2 ** 20):.2f} MB)..")
                shutil.copyfileobj(response, outfile)
            print(f"{option_name} successfully downloaded and stored at {save_path}.")
            return save_path
        except Exception as e:
            raise Exception(f"Could not download {option_name} from url {url}") from e
    else:
        raise ConfigurationException(f"Provided url {url} for {option_name} is not a valid url!")

def make_path_absolute_if_necessary(value: str, config_file_path: Path) -> str:
    """
    Convert the file path to an absolute path if it exists.

    Args:
        value (str): The file path.
        config_file_path (Path): Path to the configuration file directory.
    """
    absolute_path = (config_file_path / value).absolute()
    if absolute_path.is_file() or absolute_path.is_dir():
        return str(absolute_path)

    # File path is already absolute and not directly within the config directory
    absolute_value_path = Path(value).absolute()
    if absolute_value_path.is_file() or absolute_value_path.is_dir():
        return str(absolute_value_path)
    raise ConfigurationException(f"Could not find path for file {value}.")


def is_url(value: str) -> bool:
    """
    Check if the given value is a valid URL.

    Args:
        value (str): The value to check.

    Returns:
        bool: True if `value` is a URL with scheme http, https, or ftp, else False.
    """
    return urlparse(value).scheme in ["http", "https", "ftp"]


def is_list_option(value: Any) -> bool:
    """
    Determine if the option value is a list, range, or a valid range string.

    Used to check for hyperparameter optimization.

    Returns:
        bool: True if `value` is a list, range, or a valid range string, else False.
    """
    return ("range" in str(value) or isinstance(value, list) or
            (isinstance(value, str) and "[" in value and "]" in value))
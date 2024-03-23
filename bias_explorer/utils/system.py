"""System-wide utilities"""
from pathlib import Path


def prep_folders(path):
    """Make folders given path, if they does not exists

    :param path: directory path
    :type path: str
    """
    Path(path).mkdir(parents=True, exist_ok=True)

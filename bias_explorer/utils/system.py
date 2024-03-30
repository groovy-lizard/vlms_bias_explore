"""System-wide utilities"""
from pathlib import Path


def prep_folders(path):
    """Make folders given path, if they does not exists

    :param path: directory path
    :type path: str
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def make_out_path(conf, root):
    """Concatenate output path using conf items

    :param conf: configuration dictionary
    :type conf: dict
    :param root: root choice, must match conf key
    :type root: str
    :return: output path
    :rtype: str
    """
    root_path = conf[root]
    model_name = conf['Model']
    backbone = conf['Backbone']
    data_source = conf['DataSource']
    return f"{root_path}/{model_name}/{backbone}/{data_source}"


def grab_filename(path):
    """Grab image filename from path

    :param path: Path of image
    :type path: str
    :return: image filename
    :rtype: str
    """
    return path.split('FairFace/')[-1]

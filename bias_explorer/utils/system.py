"""System-wide utilities"""
from pathlib import Path


def prep_folders(path):
    """Make folders given path, if they does not exists

    :param path: directory path
    :type path: str
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def make_embs_path(conf):
    """Concatenate embeddings path using conf items

    :param conf: configuration dictionary
    :type conf: dict
    :return: embeddings path
    :rtype: str
    """
    embeddings_path = conf['Embeddings']
    model_name = conf['Model']
    backbone = conf['Backbone']
    data_source = conf['DataSource']
    embs_path = f"{embeddings_path}/{model_name}/{backbone}/{data_source}"
    return embs_path


def make_eval_path(conf):
    """Concatenate evaluation path using conf items

    :param conf: configuration dictionary
    :type conf: dict
    :return: evaluation path
    :rtype: str
    """
    results_path = conf['Results']
    model_name = conf['Model']
    backbone = conf['Backbone']
    data_source = conf['DataSource']
    eval_path = f"{results_path}/{model_name}/{backbone}/{data_source}"
    return eval_path


def grab_filename(path):
    """Grab image filename from path

    :param path: Path of image
    :type path: str
    :return: image filename
    :rtype: str
    """
    return path.split('FairFace/')[-1]

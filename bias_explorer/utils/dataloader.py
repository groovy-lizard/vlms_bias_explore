"""Data loading utility functions"""

import json
from glob import glob


def load_txts(path):
    """Loads text labels from path

    :param path: path to json file
    :type path: str
    :return: prompts and labels from json file
    :rtype: tuple
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    prompts = list(data.values())
    labels = list(data.keys())
    return (prompts, labels)


def load_imgs(path):
    """Load FairFace images from path

    :param path: path to image folder
    :type path: str
    :return: list of image pathnames
    :rtype: list
    """
    return glob(path + '*.jpg')

"""Data loading utility functions"""

import json
from glob import glob
import pandas as pd
import torch


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


def load_embs(img_path, txt_path):
    """Load image and text embeddings

    :param img_path: image embeddings filepath
    :type img_path: str
    :param txt_path: text embeddings filepath
    :type txt_path: str
    :return: tuple with image and text embeddings
    :rtype: tuple
    """
    img_embs = pd.read_pickle(img_path)
    txt_embs = torch.load(txt_path)
    return img_embs, txt_embs


def load_df(df_path):
    """Load pandas dataframe

    :param df_path: path to dataframe
    :type df_path: str
    :return: pandas dataframe loaded from memory
    :rtype: pd.DataFrame
    """
    return pd.read_csv(df_path)

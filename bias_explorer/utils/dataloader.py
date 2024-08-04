"""Data loading utility functions"""

import json
from glob import glob
import pandas as pd
import torch
from .. import operations
from .. import models


def load_operations():
    """Load operations modules

    :return: dictionary with operations modules loaded
    :rtype: dict[obj]
    """
    ops = {}
    ops['Generate'] = operations.generate
    ops['Predict'] = operations.predict
    ops['Report'] = operations.report
    ops['Concatenate'] = operations.concatenate
    ops['Save_imgs'] = operations.save_imgs
    ops['Analyze'] = operations.analyze
    return ops


def load_model(conf):
    """Load model based on model type, backbone and datasource

    :param conf: configuration dictionary
    :type conf: dict
    :return: an object with model, preprocessing, tokenizer and device
    :rtype: dict
    """
    if conf['Model'] == "CLIP":
        model_dict = models.clip_model.model_setup(model_name=conf['Backbone'])
    elif conf['Model'] == "openCLIP":
        model_dict = models.open_clip_model.model_setup(
            model_name=conf['Backbone'], data_source=conf['DataSource'])
    return model_dict


def load_json(path):
    """Load json from path and returns its data

    :param path: json filepath
    :type path: str
    :return: json file from path
    :rtype: dict
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_imgs(path):
    """Load FairFace images from path

    :param path: path to image folder
    :type path: str
    :return: list of image pathnames
    :rtype: list
    """
    im_list = glob(path + "*.jpg")
    if im_list == 0:
        raise ValueError("No image found in folder, check path or image type")

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
    """Load csv file as pandas dataframe

    :param df_path: path to .csv file
    :type df_path: str
    :return: pandas dataframe loaded from memory
    :rtype: pd.DataFrame
    """
    return pd.read_csv(df_path)


def generate_final_df(fface_df, scores_df):
    """Join the winning class df with the original dataset df

    :param fface_df: fairFace dataset df
    :type fface_df: pd.DataFrame
    :param scores_df: predictions df
    :type scores_df: pd.DataFrame
    :return: fairFace dataset df with new 'prediction' column
    :rtype: pd.DataFrame
    """
    new_df = fface_df.set_index('file').join(scores_df.set_index('file'))
    new_df.drop(columns=['service_test'], inplace=True)
    return new_df


def save_df(df, out):
    """Save final dataframe to out folder as csv"""
    print(f'Saving dataframe to {out}')
    df.to_csv(out)
    print('Done!')

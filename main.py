"""Main script for runing the bias_explorer package
"""

import json
from bias_explorer.models import clip_model
from bias_explorer.models import open_clip_model


def load_config(json_path="./conf.json"):
    """Load configuration file

    :param json_path: config file path, defaults to "./conf.json"
    :type json_path: str, optional
    :return: configuration file
    :rtype: dict
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_model(conf):
    """Load model based on model type, backbone and datasource

    :param conf: configuration dictionary
    :type conf: dict
    :return: an object with model, preprocessing, tokenizer and device
    :rtype: dict
    """
    if conf['Model'] == "CLIP":
        model_dict = clip_model.model_setup(model_name=conf['Backbone'])
    elif conf['Model'] == "openCLIP":
        model_dict = open_clip_model.model_setup(
            model_name=conf['Backbone'], data_source=conf['DataSource'])
    return model_dict


def main():
    conf = load_config()
    model_dict = load_model(conf)
    print(model_dict)


if __name__ == "__main__":
    main()

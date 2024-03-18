"""Main script for runing the bias_explorer package
"""

import json
from bias_explorer import operations
from bias_explorer import models


def load_config(json_path='./conf.json'):
    """Load configuration file

    :param json_path: config file path, defaults to "./conf.json"
    :type json_path: str, optional
    :return: configuration file
    :rtype: dict
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_operations():
    """Load operations modules

    :return: dictionary with operations modules loaded
    :rtype: dict[obj]
    """
    ops = {}
    ops['Generate'] = operations.generate
    ops['Evaluate'] = operations.evaluate
    ops['Report'] = operations.report
    ops['Save_imgs'] = operations.save_imgs
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


def main():
    """Reads config file and call the appropriate operations
    """
    conf = load_config()
    ops = load_operations()
    model = load_model(conf)
    for operation in conf['Operations'].items():
        if operation[1]:
            ops[operation[0]].run(conf, model)


if __name__ == "__main__":
    main()

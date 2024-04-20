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


def grab_label_name(label_filename):
    """Grab the label name from the full filename

    :param label_filename: the label filename from conf
    :type label_filename: str
    :return: the label name split from filename
    :rtype: str
    """
    label_name = label_filename.split('/')[-1].split('.')[0]
    return label_name


def list_item_swap(item_list, i1, i2):
    """Swap item_list items given value

    :param item_list: item_list containing the items
    :type item_list: list
    :param i1: value of item 1
    :type i1: any
    :param i2: value of item 2
    :type i2: any
    :return: swapped list
    :rtype: list
    """
    i = item_list.index(i1)
    j = item_list.index(i2)
    item_list[i], item_list[j] = item_list[j], item_list[i]
    return item_list


def fix_age_order(age_list):
    """Correctly sort ages list

    :param age_list: list of ages from df uniques
    :type age_list: list
    :return: new sorted age list
    :rtype: list
    """
    age_list[0] = '03-09'
    age_list[-1] = '00-02'
    age_list.sort()
    return age_list

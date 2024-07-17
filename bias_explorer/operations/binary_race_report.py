"""Report generator module"""
import glob
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pandas as pd
from ..utils import system
from ..utils import dataloader


def filter_df(df, col, val):
    """Filter dataframe by column and value"""
    return df[df[col] == val]


def accuracy_eval(df, metric):
    """Return the accuracy score of the true label vs predictions"""
    return round(metric(df['race'], df['race_preds']), 4)


def metric_loader(metric_name):
    """Loads the corresponding metric function"""
    metric_functions = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score
    }
    return metric_functions[metric_name]


def acc_by_col(df, col, metric, writer):
    """Generate predictions accuracy by column"""
    for unique in df[col].unique():
        print(f"Measuring {unique}...")
        col_df = filter_df(df, col, unique)
        col_acc = accuracy_eval(col_df, metric)
        writer.write(f"{unique} predictions accuracy: {round(col_acc, 4)} \n")


def get_topk_preds(preds_path):
    """Grab the path of all top 'k' predictions

    :param preds_path: root prediction path
    :type preds_path: str
    :return: a dictionary with k: pred_path
    :rtype: dict
    """
    preds_list = sorted(glob.glob(preds_path + '/binary_race_top*.csv'))
    k_preds = {}
    for pred in preds_list:
        fname = pred.split("/")[-1]
        name_parts = fname.split("_")
        k = name_parts[2]
        k_preds[k] = pred
    return k_preds


def get_col_list(df, metric_name):
    """Get the list of columns from a given dataframe

    :param df: dataframe with predictions
    :type df: pd.DataFrame
    :param metric_name: name of the metric to be used
    :type metric_name: str
    :return: list of columns from given df
    :rtype: list
    """
    if metric_name == "balanced_accuracy":
        col_list = list(df.drop(
            columns=['file', 'race_preds', 'race']).keys())
    else:
        col_list = list(df.drop(columns=['file', 'race_preds']).keys())
        col_list = system.list_item_swap(col_list, 'age', 'gender')
    col_list = system.list_item_swap(col_list, 'age', 'gender')
    return col_list


def get_uniques(df, col_list):
    """Get Unique values from columns

    :param df: dataframe with predictions
    :type df: pd.DataFrame
    :param col_list: list of columns from dataframe
    :type col_list: list
    :return: list of unique values from column
    :rtype: list
    """
    uniques = []
    for col in col_list:
        col_items = list(df[col].unique())
        if col == "age":
            col_items = system.fix_age_order(col_items)
        for unique in col_items:
            if col == "age" and unique == "00-02":
                unique = "0-2"
            elif col == "age" and unique == "03-09":
                unique = "3-9"
            uniques.append(unique)
    return uniques


def get_empty_report_dict(df, metric_name):
    """Generate a new empty report dictionary

    :param df: dataframe with predictions to extract columns
    :type df: pd.DataFrame
    :param metric_name: name of the metric to be used
    :type metric_name: str
    :return: an empty report dictionary
    :rtype: dict
    """
    report_dict = {
        'Mode': [],
        metric_name: [],
    }

    col_list = get_col_list(df, metric_name)
    uniques = get_uniques(df, col_list)
    for unique in uniques:
        report_dict[unique] = []
    return report_dict


def gen_dict_report(df, mode_name, metric_name, rep_dict):
    """Generate the dictionary report

    :param df: dataframe with predictions
    :type df: pd.DataFrame
    :param mode_name: name of the prediction mode used
    :type mode_name: str
    :param metric_name: name of the metric used
    :type metric_name: str
    :param rep_dict: current report dictionary
    :type rep_dict: str
    :return: updated report dictionary
    :rtype: dict
    """
    metric_func = metric_loader(metric_name)
    rep_dict['Mode'].append(mode_name)
    rep_dict[metric_name].append(accuracy_eval(df, metric_func))
    col_list = get_col_list(df, metric_name)
    for col in col_list:
        uniques = get_uniques(df, [col])
        for unique in uniques:
            col_df = filter_df(df, col, unique)
            col_acc = accuracy_eval(col_df, metric_func)
            rep_dict[unique].append(col_acc)
    return rep_dict


def add_topk_preds(preds_path, metric_name, rep_dict):
    """Add the report of the top k predictions

    :param preds_path: path to root predictions folder
    :type preds_path: str
    :param metric_name: name of the metric used
    :type metric_name: str
    :param rep_dict: current dictionary to add top k reports
    :type rep_dict: dict
    :return: updated reports dictionary
    :rtype: dict
    """
    k_preds = get_topk_preds(preds_path)
    for k, path in k_preds.items():
        mode_name = f"Top {k}"
        k_df = dataloader.load_df(path)
        rep_dict = gen_dict_report(k_df, mode_name, metric_name, rep_dict)
    return rep_dict


def run(conf):
    """Run report generator

    :param conf: configuration dictionary
    :type conf: dict
    :param _: unused model parameter
    :type _: None
    """
    print("Generating Report...")
    metric_name = conf['Metric']
    preds = system.concat_out_path(conf, 'Predictions')
    report_path = system.concat_out_path(conf, 'Reports')
    system.prep_folders(report_path)
    out_csv = f"{report_path}/{metric_name}_report.csv"

    sum_df = dataloader.load_df(f"{preds}/binary_race_avg_preds.csv")

    rep_dict = get_empty_report_dict(sum_df, metric_name)
    rep_dict = gen_dict_report(sum_df, "Avg Sum", metric_name, rep_dict)
    rep_dict = add_topk_preds(preds, metric_name, rep_dict)

    rep_df = pd.DataFrame(rep_dict)
    rep_df.to_csv(out_csv)
    print("Done!")

"""Report generator module"""
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pandas as pd
from ..utils import system
from ..utils import dataloader


def filter_df(df, col, val):
    """Filter dataframe by column and value

    :param df: dataframe to be filtered
    :type df: pd.DataFrame
    :param col: target column
    :type col: str
    :param val: filtering value
    :type val: str
    :return: filtered dataframe
    :rtype: pd.DataFrame
    """
    return df[df[col] == val]


def gender_eval(df, metric):
    """Return the accuracy score of the gender vs gender predictions

    :param df: dataframe with true label gender and gender preds
    :type df: pd.DataFrame
    :return: accuracy score of predictions
    :rtype: float
    """
    return metric(df['gender'], df['gender_preds'])


def metric_loader(metric_name):
    """Loads the corresponding metric function

    :param metric_name: name of the metric set on config file
    :type metric_name: str
    :return: function of the chosen metric
    :rtype: func
    """
    metric_functions = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score
    }
    return metric_functions[metric_name]


def acc_by_col(df, col, metric, writer):
    """Generate predictions accuracy by column

    :param df: target dataframe
    :type df: pd.DataFrame
    :param col: target column
    :type col: str
    :param writer: file writer object
    :type writer: io.TextIOWrapper
    """
    for unique in df[col].unique():
        col_df = filter_df(df, col, unique)
        col_acc = gender_eval(col_df, metric)
        writer.write(f"{unique} predictions accuracy: {round(col_acc, 5)} \n")


def gen_txt_report(df, metric, pred_name, label_name, out):
    """Generate textual report

    :param df: predictions dataframe
    :type df: pd.DataFrame
    :param pred_name: prediction mode
    :type pred_name: str
    :param label_name: label mode
    :type label_name: str
    :param out: output folder for the report file
    :type out: str
    """
    print(f"Generating {pred_name} txt report...")
    with open(out, mode="w", encoding="utf-8") as fp:
        fp.write("# Final Report \n")
        fp.write(f"\nPrediction mode: {pred_name}\n")
        fp.write(f"Label mode: {label_name} \n")
        fp.write("-"*20)

        fp.write("\n## General Accuracy\n")
        gen_acc = gender_eval(df, metric)
        gen_miss = df[df['gender'] != df['gender_preds']]
        fp.write(f"Prediction error count: {len(gen_miss)} \n")
        fp.write(f"Prediction accuracy score: {round(gen_acc, 5)} \n")

        fp.write("\n## Accuracy by gender \n")
        male_df = filter_df(df, 'gender', 'Male')
        male_acc = gender_eval(male_df, metric)
        female_df = filter_df(df, 'gender', 'Female')
        female_acc = gender_eval(female_df, metric)
        fp.write(f"Male: {round(male_acc, 5)} \n")
        fp.write(f"Female: {round(female_acc, 5)} \n")

        fp.write("\n## Accuracy by race \n")
        acc_by_col(df, 'race', metric, fp)

        fp.write("\n## Accuracy by age \n")
        acc_by_col(df, 'age', metric, fp)
        fp.write("-"*20)
    print("Saved at " + out)


def gen_csv_report(sum_df, top_df, metric, out):
    """Generate final csv report

    :param sum_df: average sum dataframe
    :type sum_df: pd.DataFrame
    :param top_df: top 1 dataframe
    :type top_df: pd.DataFrame
    :param out: output filepath
    :type out: str
    """
    print("Generating csv report...")
    rep_dict = {}
    rep_dict['Mode'] = ['Avg Sum', 'Top 1']
    rep_dict['Accuracy'] = [gender_eval(
        sum_df, metric), gender_eval(top_df, metric)]

    col_list = list(sum_df.drop(columns=['file', 'gender_preds']).keys())
    col_list = system.list_item_swap(col_list, 'age', 'gender')
    col_list = system.list_item_swap(col_list, 'age', 'race')

    for col in col_list:
        col_items = list(sum_df[col].unique())
        if col == "age":
            col_items = system.fix_age_order(col_items)
        for unique in col_items:
            if col == "age" and unique == "00-02":
                data_unique = "0-2"
            elif col == "age" and unique == "03-09":
                data_unique = "3-9"
            else:
                data_unique = unique
            sum_col_df = filter_df(sum_df, col, data_unique)
            sum_col_acc = gender_eval(sum_col_df, metric)
            top_col_df = filter_df(top_df, col, data_unique)
            top_col_acc = gender_eval(top_col_df, metric)
            rep_dict[unique] = [sum_col_acc, top_col_acc]
    rep_df = pd.DataFrame(rep_dict)
    rep_df.to_csv(out)
    print("Saved at " + out)


def run(conf):
    """Run report generator

    :param conf: configuration dictionary
    :type conf: dict
    :param _: unused model parameter
    :type _: None
    """
    print("Generating Report...")
    metric_func = metric_loader(conf['Metric'])
    label_name = system.grab_label_name(conf['Labels'])
    eval_path = system.make_out_path(conf, 'Results')
    report_root_path = system.make_out_path(conf, 'Reports')
    report_path = f"{report_root_path}/{label_name}"
    system.prep_folders(report_path)

    out_csv = f"{report_path}/report.csv"

    sum_df = dataloader.load_df(f"{eval_path}/sum_synms.csv")
    top_df = dataloader.load_df(f"{eval_path}/top_synms.csv")

    gen_csv_report(sum_df, top_df, metric_func, out_csv)

    print("Done!")

"""Reports concatenation module"""
import os
import pandas as pd
from ..utils import dataloader, system


def run(conf):
    """Run the concatenation script

    :param conf: configuration dictionary
    :type conf: dict
    :param _: unused model
    :type _: None
    """
    model = conf['Model']
    backbone = conf['Backbone']
    metric = conf['Metric']
    report_path = conf['Reports']
    ds_path = f"{report_path}/{model}/{backbone}"
    ln = system.grab_label_name(conf['Labels'])
    out_path = f"{report_path}/{model}/{backbone}_{metric}_{ln}.csv"

    print("Collecting reports...")
    ds_list = os.listdir(ds_path)
    reports = {}
    for dsource in ds_list:
        report_path = f"{ds_path}/{dsource}/{ln}/{metric}_report.csv"
        reports[dsource] = dataloader.load_df(report_path)
        reports[dsource].drop(columns=['Unnamed: 0'], inplace=True)
    print("Concating...")
    report_df = pd.concat(reports)
    report_df.rename(
        columns={"Latino_Hispanic": "Latino Hispanic"}, inplace=True)
    print("Saving...")
    report_df.to_csv(out_path)
    print("Done!")

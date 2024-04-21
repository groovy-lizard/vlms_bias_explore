"""Reports concatenation module"""
import os
import sys
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
    root_path = conf['Reports']
    label_name = system.grab_label_name(conf['Labels'])
    ds_path = f"{root_path}/{model}/{backbone}"
    out_path = f"{root_path}/{model}/{backbone}_{label_name}.csv"

    if os.path.isfile(out_path):
        sys.exit()

    print("Collecting reports...")
    ds_list = os.listdir(ds_path)
    reports = {}
    for datasource in ds_list:
        report_path = f"{ds_path}/{datasource}/{label_name}/report.csv"
        reports[datasource] = dataloader.load_df(report_path)
        reports[datasource].drop(columns=['Unnamed: 0'], inplace=True)
    print("Concating...")
    report_df = pd.concat(reports)
    report_df.rename(
        columns={"Latino_Hispanic": "Latino Hispanic"}, inplace=True)
    print("Saving...")
    report_df.to_csv(out_path)
    print("Done!")

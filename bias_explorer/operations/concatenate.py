"""Reports concatenation module"""
import os
import pandas as pd
from ..utils import dataloader, system


def collect_reports(ds_path, ln, metric):
    """Collect reports from all datasources

    :param ds_path: path to datasources root folder
    :type ds_path: str
    :param ln: label name
    :type ln: str
    :param metric: metric name
    :type metric: str
    :return: dictionary with datasource: dataframe
    :rtype: dict
    """
    ds_list = os.listdir(ds_path)
    reports = {}
    for dsource in ds_list:
        report_path = f"{ds_path}/{dsource}/{ln}/{metric}_report.csv"
        ds_df = dataloader.load_df(report_path)
        ds_df = ds_df.drop(columns=['Unnamed: 0'])
        reports[dsource] = ds_df
    return reports


def filter_best_modes(reports):
    """Retrieve roll of the best prediciton mode

    :param reports: dictionary with reports
    :type reports: dict
    :return: a filtered report with only best row
    :rtype: dict
    """
    filtered_reports = {}
    for dsname, df in reports.items():
        filtered_reports[dsname] = df.loc[df['accuracy'].idxmax()]
    return filtered_reports


def grab_topk(reports):
    """Retrieve the top k row of reports

    :param reports: dictionary with reports
    :type reports: dict
    """
    filtered_reports = {}
    for dsname, df in reports.items():
        filtered_reports[dsname] = df.loc[df['Mode'] == 'Top 1'].squeeze()
    return filtered_reports


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
    reps = collect_reports(ds_path, ln, metric)
    best_reps = filter_best_modes(reps)
    # best_reps = grab_topk(reps)

    print("Concating...")
    report_df = pd.DataFrame(best_reps.values(), index=best_reps.keys())
    print("Saving...")
    report_df.to_csv(out_path)
    print("Done!")

"""Analyzer module to generate final tabular analytics"""
import os
import pandas as pd
from bias_explorer.utils import dataloader
from bias_explorer.utils import system


def race_gap(df: pd.DataFrame) -> pd.Series:
    """Compute race gap (mean - min)

    :param df: input dataframe
    :type df: pd.DataFrame
    :return: computed gap value
    :rtype: pd.Series
    """
    race_list = ['East Asian', 'White', 'Latino_Hispanic',
                 'Southeast Asian', 'Black', 'Indian', 'Middle Eastern']
    race_df = df[race_list]
    mins = race_df.min(axis=1)
    mean = race_df.mean(axis=1)
    return round(mean - mins, 4)


def prep_df(path: str, arch: str) -> pd.DataFrame:
    """Read and prepare dataframe

    :param path: path to csv file
    :type path: str
    :param arch: current architecture name
    :type arch: str
    :return: dataframe read from csv file with arch name
    :rtype: pd.DataFrame
    """
    df = pd.read_csv(path)
    df = df[df['Mode'] == 'Top 01']
    df = df.rename(columns={'Unnamed: 0': 'Model'})
    df['Model'] = df['Model'].map({1: arch})
    df = df.drop(columns=['Mode'])
    df = df.set_index('Model')
    return df


def get_df_list(conf: dict, label_name: str) -> pd.DataFrame:
    """get dataframe list given label name"""
    report_path = f"{conf['Reports']}/{conf['Model']}"
    target_label_folder = f"{conf['Target']}_{label_name}"
    archs_and_dsources = dataloader.load_json("./archs_and_datasources.json")
    df_list = []
    for arch, datasources in archs_and_dsources.items():
        if arch in ("ViT-H-14", "ViT-g-14"):
            ds = datasources[-1]
        else:
            ds = "openai"
        arch_ds_path = f"{report_path}/{arch}/{ds}"
        results_path = f"{arch_ds_path}/{target_label_folder}"
        results_file = results_path + "/accuracy_report.csv"
        results_df = prep_df(results_file, arch)
        rgap = race_gap(results_df)
        results_df['Race Gap'] = rgap
        if label_name == "age_race_gender":
            results_df['Prompt'] = "RAGP"
        elif label_name == "original_clip_labels":
            results_df['Prompt'] = "RGP"
        else:
            results_df['Prompt'] = "RP"
        df_list.append(results_df)
    df = pd.concat(df_list)
    return df


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


def grab_top_01(reports):
    """Retrieve the top k row of reports

    :param reports: dictionary with reports
    :type reports: dict
    """
    filtered_reports = {}
    for dsname, df in reports.items():
        filtered_reports[dsname] = df.loc[df['Mode'] == 'Top 01'].squeeze()
    return filtered_reports


def model_scaling_analysis(conf: dict):
    """Analyze model scaling by grabbing all model results
    for laion2B datasource"""
    print("Performing model scaling analysis...")
    report_path = f"{conf['Reports']}/{conf['Model']}"
    label_name = system.grab_label_name(conf['Labels'])
    target_label_folder = f"{conf['Target']}_{label_name}"
    archs_and_dsources = dataloader.load_json("./archs_and_datasources.json")
    df_list = []
    for arch, datasources in archs_and_dsources.items():
        if arch == "ViT-L-14":
            laion2b = datasources[-2]
        else:
            laion2b = datasources[-1]
        arch_ds_path = f"{report_path}/{arch}/{laion2b}"
        results_path = f"{arch_ds_path}/{target_label_folder}"
        results_file = results_path + "/accuracy_report.csv"
        results_df = prep_df(results_file, arch)
        rgap = race_gap(results_df)
        results_df['Race Gap'] = rgap
        df_list.append(results_df)
    df = pd.concat(df_list)
    root_path = f"{conf['EDA']}/{conf['Target']}"
    out_path = f"{root_path}_classification/{label_name}/model_scaling.csv"
    print(f"Saving to {out_path}")
    df.to_csv(out_path)
    print("Done!")


def prompt_analysis(conf: dict):
    """Compare prompt strategies (table VI)"""
    print("Performing prompt comparison analysis...")
    ragp_df = get_df_list(conf, "age_race_gender")
    rgp_df = get_df_list(conf, "original_clip_labels")
    rp_df = get_df_list(conf, "raw_race_labels")
    final_df = pd.concat([ragp_df, rgp_df, rp_df])
    final_df = final_df.sort_values('Model', ascending=True)
    root_path = f"{conf['EDA']}/{conf['Target']}"
    out_path = f"{root_path}_classification/prompt_comparison.csv"
    print(f"Saving to {out_path}")
    final_df.to_csv(out_path)
    print("Done!")


def data_scaling_analysis(conf):
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
    label = system.grab_label_name(conf['Labels'])
    ln = f"{conf['Target']}_{label}"
    final_path = f"{conf['EDA']}/{conf['Target']}_classification/{label}"
    out_path = f"{final_path}/{backbone}_data_scaling.csv"

    print("Collecting reports...")
    reps = collect_reports(ds_path, ln, metric)
    best_reps = grab_top_01(reps)

    print("Generating data scaling analysis...")
    report_df = pd.DataFrame(best_reps.values(), index=best_reps.keys())
    report_df = report_df.drop(columns=['Mode'])
    report_df['Race Gap'] = race_gap(report_df)
    print(f"Saving to {out_path}")
    report_df.to_csv(out_path, index_label="datasource")
    print("Done!")


def top_k_analysis(conf):
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
    label = system.grab_label_name(conf['Labels'])
    ln = f"{conf['Target']}_{label}"
    final_path = f"{conf['EDA']}/{conf['Target']}_classification/{label}"
    out_path = f"{final_path}/{backbone}_top_k_analysis.csv"

    print("Collecting reports...")
    reps = collect_reports(ds_path, ln, metric)
    best_reps = filter_best_modes(reps)

    print("Generating Top K analysis...")
    report_df = pd.DataFrame(best_reps.values(), index=best_reps.keys())
    report_df['Race Gap'] = race_gap(report_df)
    print(f"Saving to {out_path}")
    report_df.to_csv(out_path, index_label="datasource")
    print("Done!")


def prep_eda_folders(conf):
    """Create EDA folder tree if not exists"""
    label = system.grab_label_name(conf['Labels'])
    final_path = f"{conf['EDA']}/{conf['Target']}_classification/{label}"
    system.prep_folders(final_path)


def run(conf):
    """Run the analyzer module"""
    prep_eda_folders(conf)
    model_scaling_analysis(conf)
    prompt_analysis(conf)
    data_scaling_analysis(conf)
    top_k_analysis(conf)

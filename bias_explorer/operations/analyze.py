"""Analyzer module to generate final tabular analytics"""
import pandas as pd
from bias_explorer.utils import dataloader, system


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
    df = df[df['Mode'] == 'Top 1']
    df = df.rename(columns={'Unnamed: 0': 'Model'})
    df['Model'] = df['Model'].map({1: arch})
    df = df.drop(columns=['Mode'])
    df = df.set_index('Model')
    return df


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
    df.to_csv(
        f"{conf['EDA']}/{conf['Target']}_classification/model_scaling.csv")
    print("Done!")


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
        else:
            results_df['Prompt'] = "RGP"
        df_list.append(results_df)
    df = pd.concat(df_list)
    return df


def prompt_analysis(conf: dict):
    """Compare prompt strategies (table VI)"""
    print("Performing prompt comparison analysis...")
    ragp_df = get_df_list(conf, "age_race_gender")
    rgp_df = get_df_list(conf, "original_clip_labels")
    final_df = pd.concat([ragp_df, rgp_df])
    final_df = final_df.sort_values('Model', ascending=True)
    final_df.to_csv(
        f"{conf['EDA']}/{conf['Target']}_classification/prompt_comparison.csv")
    print("Done!")


def run(conf):
    """Run the analyzer module"""
    model_scaling_analysis(conf)
    prompt_analysis(conf)

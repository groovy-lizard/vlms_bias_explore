"""Report generator module"""
from sklearn.metrics import accuracy_score as accuracy_score
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


def gender_acc(df):
    """Return the accuracy score of the gender vs gender predictions

    :param df: dataframe with true label gender and gender preds
    :type df: pd.DataFrame
    :return: accuracy score of predictions
    :rtype: float
    """
    return accuracy_score(df['gender'], df['gender_preds'])


def acc_by_col(df, col, writer):
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
        col_acc = gender_acc(col_df)
        writer.write(f"{unique} predictions accuracy: {round(col_acc, 5)} \n")


def gen_report(df, pred_name, label_name, out):
    """Wrapper report class

    :param df: predictions dataframe
    :type df: pd.DataFrame
    :param pred_name: prediction mode
    :type pred_name: str
    :param label_name: label mode
    :type label_name: str
    :param out: output folder for the report file
    :type out: str
    """
    with open(out, mode="w", encoding="utf-8") as fp:
        fp.write("# Final Report \n")
        fp.write(f"\nPrediction mode: {pred_name}\n")
        fp.write(f"Label mode: {label_name} \n")
        fp.write("-"*20)

        fp.write("\n## General Accuracy\n")
        gen_acc = gender_acc(df)
        gen_miss = df[df['gender'] != df['gender_preds']]
        fp.write(f"Prediction error count: {len(gen_miss)} \n")
        fp.write(f"Prediction accuracy score: {round(gen_acc, 5)} \n")

        fp.write("\n## Accuracy by gender \n")
        male_df = filter_df(df, 'gender', 'Male')
        male_acc = gender_acc(male_df)
        female_df = filter_df(df, 'gender', 'Female')
        female_acc = gender_acc(female_df)
        fp.write(f"Male: {round(male_acc, 5)} \n")
        fp.write(f"Female: {round(female_acc, 5)} \n")

        fp.write("\n## Accuracy by race \n")
        acc_by_col(df, 'race', fp)

        fp.write("\n## Accuracy by age \n")
        acc_by_col(df, 'age', fp)
        fp.write("-"*20)


def run(conf, _):
    """Run report generator

    :param conf: configuration dictionary
    :type conf: dict
    :param _: unused model parameter
    :type _: None
    """
    print("Generating Report...")
    label_name = system.grab_label_name(conf['Labels'])
    eval_path = system.make_out_path(conf, 'Results')
    report_root_path = system.make_out_path(conf, 'Reports')
    report_path = f"{report_root_path}/{label_name}"
    system.prep_folders(report_path)

    out_sum = f"{report_path}/sum_report.txt"
    out_top = f"{report_path}/top_report.txt"

    sum_df = dataloader.load_df(f"{eval_path}/sum_synms.csv")
    top_df = dataloader.load_df(f"{eval_path}/top_synms.csv")

    gen_report(sum_df, "Average Sum", label_name, out_sum)
    gen_report(top_df, "Top K", label_name, out_top)

    print("Done!")

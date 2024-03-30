"""Report generator module"""
from sklearn.metrics import accuracy_score
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
        writer.write(f"{unique} predictions accuracy: {round(col_acc, 2)} \n")


def gen_report(df, pred_name, out):
    """Wrapper report class

    :param df: predictions dataframe
    :type df: pd.DataFrame
    :param pred_name: name of the report
    :type pred_name: str
    :param out: output folder for the report file
    :type out: str
    """
    fp = open(out, mode="a", encoding="utf-8")
    fp.write(f"\n# {pred_name} info:\n")
    fp.write("-"*20)

    fp.write("\n## General Accuracy\n")
    gen_acc = gender_acc(df)
    gen_miss = df[df['gender'] != df['gender_preds']]
    fp.write(f"Prediction error count: {len(gen_miss)} \n")
    fp.write(f"Prediction accuracy score: {round(gen_acc, 2)} \n")

    fp.write("\n## Accuracy by gender \n")
    male_df = filter_df(df, 'gender', 'Male')
    male_acc = gender_acc(male_df)
    female_df = filter_df(df, 'gender', 'Female')
    female_acc = gender_acc(female_df)
    fp.write(f"Male: {round(male_acc, 2)} \n")
    fp.write(f"Female: {round(female_acc, 2)} \n")

    fp.write("\n## Accuracy by race \n")
    acc_by_col(df, 'race', fp)

    fp.write("\n## Accuracy by age \n")
    acc_by_col(df, 'age', fp)
    fp.write("-"*20)
    fp.close()


def run(conf, _):
    """Run report generator

    :param conf: configuration dictionary
    :type conf: dict
    :param _: unused model parameter
    :type _: None
    """
    print("Generating Report...")
    eval_path = system.make_out_path(conf, 'Results')
    report_path = system.make_out_path(conf, 'Reports')
    system.prep_folders(report_path)
    out_sum = f"{report_path}/sum_report.txt"
    out_top = f"{report_path}/top_report.txt"
    sum_df = dataloader.load_df(f"{eval_path}/sum_synms.csv")
    top_df = dataloader.load_df(f"{eval_path}/top_synms.csv")
    gen_report(sum_df, "Average Sum Predictions", out_sum)
    gen_report(top_df, "Top K Predictions", out_top)
    print("Done!")

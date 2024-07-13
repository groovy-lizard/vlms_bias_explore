"""Evaluation module for comparisons between text and image embeddings"""
from . import gender_report, race_report


def run(conf):
    """Run the Evaluator module

    :param conf: config file
    :type conf: dict
    :param model: model utilities object
    :type model: dict[obj]
    """
    print("Initializing reporters...")
    reporters = {"Gender": gender_report,
                 "Race": race_report,
                 "Age": 1}
    reporters[conf['Target']].run(conf)

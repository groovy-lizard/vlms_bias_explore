"""Evaluation module for comparisons between text and image embeddings"""
from . import gender_predict


def run(conf):
    """Run the Evaluator module

    :param conf: config file
    :type conf: dict
    :param model: model utilities object
    :type model: dict[obj]
    """
    print("Initializing predictors...")
    predictors = {"Gender": gender_predict,
                  "Race": 1,
                  "Age": 1}
    predictors[conf['Target']].run(conf)

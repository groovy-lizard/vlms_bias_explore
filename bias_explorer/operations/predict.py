"""Evaluation module for comparisons between text and image embeddings"""
from . import gender_predict
from . import race_predict
from . import binary_race_predict


def run(conf):
    """Run the Evaluator module

    :param conf: config file
    :type conf: dict
    :param model: model utilities object
    :type model: dict[obj]
    """
    print("Initializing predictors...")
    predictors = {"Gender": gender_predict,
                  "Race": race_predict,
                  "Binary Race": binary_race_predict,
                  "Age": 1}
    predictors[conf['Target']].run(conf)

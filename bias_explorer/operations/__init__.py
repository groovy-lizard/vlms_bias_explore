"""Operations package"""
from . import generate, predict, report, concatenate, save_imgs, analyze


def __init__(self):
    self.generate = generate
    self.predict = predict
    self.report = report
    self.concatenate = concatenate
    self.save_imgs = save_imgs
    self.analyze = analyze

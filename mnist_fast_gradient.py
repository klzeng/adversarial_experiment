# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 16:02:23 2018

@author: fzp
"""

from keras.models import load_model
import numpy as np


model = load_model()


def preprocessing(data):
    return data / 255.

def recover_from_preprocessing(data):
    return data * 255.

def get_predict_label(X, model):
    return np.argmax(model.predict(X), axis=1)


if

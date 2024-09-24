import pickle

import numpy as np
from sklearn.linear_model import LinearRegression


def get_model():
    """
    Currently use baseline model for good predictions
    :return: Model
    """
    model = pickle.load(open('data/model_average.pkl', 'rb'))
    return model


def predict_traffic(value, model):
    weights = [1] * 10
    values = [value[f'{i}'] for i in range(10)]
    average = np.average(values, weights=weights)
    inputs = [average]
    result = model.predict([inputs])
    return result[0]

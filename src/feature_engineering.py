import pickle

import matplotlib.pyplot as plt
import numpy as np
from dateutil.rrule import weekday
from numpy import ndarray, shape

from src.util import get_data_from_pickle, create_regression_model

if __name__ == '__main__':
    # Load datasets
    tra_X_tr = get_data_from_pickle('tra_X_tr')
    tra_Y_tr = get_data_from_pickle('tra_Y_tr')

    # Currently, limit to one location
    location_index = 0

    # This is what we want to predict
    output_data = tra_Y_tr[location_index, :]

    input_data: ndarray = tra_X_tr[0]
    input_data_size = len(input_data)
    feature_list = np.empty(shape=(input_data_size, 48))

    for i in range(input_data_size):
        # we need data for all timestamps
        timestamp_data = input_data.item(i).toarray()
        for j in range(48):
            feature_list[i, j] = timestamp_data[0][j]

    '''
    We are using three different approaches here
    1. Historical Data
    2. Week and Time of Day
    3. All Data Combined
    '''
    feature_split = [slice(i, i + 1) for i in range(9)] \
                    + [slice(9, 41)] \
                    + [slice(0, -1)]

    y_train = output_data

    for item in feature_split:
        x_train = feature_list[:, item]
        model = create_regression_model(x_train, y_train, f'feature{item.start}_{item.stop}')

    '''
    Lets engineer new features now!
    1. Historical Data as rolling average
    2. Historical Data as weighted average
    3. Time & Date as a single feature i.e. 0-24 as 0-1 & weekdays as 0-1
    '''
    # Reverse Average
    weights = [i for i in range(10)]
    features = feature_list[:, 0:10]
    weighted_average = np.average(features, weights=weights, axis=1)
    x_train = weighted_average[:, None]
    create_regression_model(x_train, y_train, 'reverse_weighted_average')

    # Weighted Average
    weights = [10 - i for i in range(10)]
    features = feature_list[:, 0:10]
    weighted_average = np.average(features, weights=weights, axis=1)
    x_train = weighted_average[:, None]
    create_regression_model(x_train, y_train, 'weighted_average')

    # Average
    weights = [1 for i in range(10)]
    features = feature_list[:, 0:10]
    weighted_average = np.average(features, weights=weights, axis=1)
    x_train = weighted_average[:, None]
    create_regression_model(x_train, y_train, 'average')


    # Week on linear scale
    features = feature_list[:, slice(10, 17)]
    weighted_average_week = features.argmax(axis=1)
    weighted_average_week = np.divide(weighted_average_week, 7)
    x_train = weighted_average_week[:, None]
    create_regression_model(x_train, y_train, 'week')

    # Day on a linear scale
    weights = [i for i in range(24)]
    features = feature_list[:, 16:40]
    weighted_average_day = np.average(features, weights=weights, axis=1)
    x_train = weighted_average_day[:, None]
    create_regression_model(x_train, y_train, 'day')

    #Final
    weights = [i for i in range(24)]
    x_train = feature_list[:, 10:40] + np.array([weighted_average]).transpose()
    create_regression_model(x_train, y_train, 'time_and_average')

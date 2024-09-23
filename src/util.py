import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.read_data import get_file_name


def get_data_from_pickle(pickle_file):
    with open(get_file_name(pickle_file), 'rb') as f:
        data = pickle.load(f)
    return data


def create_regression_model(x_train, y_train, feature_name, plot=False):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    mae = mean_absolute_error(y_train, y_train_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"Training MAE: {mae}")
    print(f"Training RMSE: {rmse}")

    with open(f'data/model_{feature_name}.pkl', 'wb') as _:
        pickle.dump(model, _)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(y_train)
        plt.plot(x_train[:, 0])
        plt.title(f"Traffic Flow predicted by feature {feature_name}")
        plt.xlabel("Time (15-minute intervals)")
        plt.ylabel("Traffic Flow")
        plt.savefig(f"Prediction with all features.jpg")

    return mae, rmse

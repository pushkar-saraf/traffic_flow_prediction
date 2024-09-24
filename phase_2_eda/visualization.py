# Load the training data from the pickle files
import pickle

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.read_data import get_file_name

'''
Get all data
'''
with open(get_file_name('tra_X_tr'), 'rb') as f:
    tra_X_tr = pickle.load(f)

with open(get_file_name('tra_Y_tr'), 'rb') as f:
    tra_Y_tr = pickle.load(f)

with open(get_file_name('tra_adj_mat'), 'rb') as f:
    tra_adj_mat = pickle.load(f)




'''
Plot some basic stuff
'''

location_index = 0
traffic_flow_series = tra_Y_tr[location_index]
plt.figure(figsize=(10, 5))
plt.plot(traffic_flow_series)
plt.title(f"Traffic Flow Over Time at Location {location_index + 1}")
plt.xlabel("Time (15-minute intervals)")
plt.ylabel("Traffic Flow")
plt.show()
#
# # Step 2: Autocorrelation and Partial Autocorrelation
plot_acf(traffic_flow_series, lags=50)
plt.title(f"Autocorrelation Function (ACF) for Location {location_index + 1}")
plt.legend()
plt.show()
#
plot_pacf(traffic_flow_series, lags=50)
plt.title(f"Partial Autocorrelation Function (PACF) for Location {location_index + 1}")
plt.show()

plt.hist(traffic_flow_series, density=False, bins=50)
plt.xlabel('Traffic Flow')
plt.ylabel('Count')
plt.show()

'''
Plot additional time series info and features
'''

location_index = 0
output_data = tra_Y_tr[location_index, :]

input_data: ndarray = tra_X_tr[0]
input_data_size = len(input_data)
feature_list = np.empty(shape=(input_data_size, 48))

for i in range(input_data_size):
    # we need data for all timestamps
    timestamp_data = input_data.item(i).toarray()
    for j in range(48):
        feature_list[i, j] = timestamp_data[0][j]

feature_split = [slice(i, i + 1) for i in range(9)] + [slice(9, 41)]

for item in feature_split:
    model = LinearRegression()
    x_train = feature_list[:, item]
    y_train = output_data
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    mae = mean_absolute_error(y_train, y_train_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"Training MAE: {mae}")
    print(f"Training RMSE: {rmse}")

    with open(f'data/model{item.start + 1}.pkl', 'wb') as _:
        pickle.dump(model, _)

model = LinearRegression()
x_train = feature_list
y_train = output_data
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
mae = mean_absolute_error(y_train, y_train_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print(f"Training MAE: {mae}")
print(f"Training RMSE: {rmse}")

# save
with open('data/model.pkl', 'wb') as _:
    pickle.dump(model, _)

# plt.figure(figsize=(10, 5))
# plt.plot(traffic_flow_series)
# plt.title(f"Traffic Flow Over Time at Location {location_index + 1}")
# plt.xlabel("Time (15-minute intervals)")
# plt.ylabel("Traffic Flow")
# plt.show()
#
# # Step 2: Autocorrelation and Partial Autocorrelation
plot_acf(traffic_flow_series, lags=50)
plt.title(f"Autocorrelation Function (ACF) for Location {location_index + 1}")
plt.legend()
plt.show()
#
plot_pacf(traffic_flow_series, lags=50)
plt.title(f"Partial Autocorrelation Function (PACF) for Location {location_index + 1}")
plt.show()

plt.hist(traffic_flow_series, density=False, bins=50)
plt.xlabel('Traffic Flow')
plt.ylabel('Count')
plt.show()

location_index = 0
output_data = tra_Y_tr[location_index, :]

input_data: ndarray = tra_X_tr[0]
input_data_size = len(input_data)
feature_list = np.empty(shape=(input_data_size, 48))

for i in range(input_data_size):
# we need data for all timestamps
    timestamp_data: array = input_data.item(i).toarray()
for j in range(48):
    feature_list[i, j] = timestamp_data[0][j]

feature_split = [slice(i, i + 1) for i in range(9)] + [slice(9, 41)]

for item in feature_split:
    model = LinearRegression()
x_train = feature_list[:, item]
y_train = output_data
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
mae = mean_absolute_error(y_train, y_train_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print(f"Training MAE: {mae}")
print(f"Training RMSE: {rmse}")

with open(f'data/model{item.start + 1}.pkl', 'wb') as _:
    pickle.dump(model, _)

model = LinearRegression()
x_train = feature_list
y_train = output_data
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
mae = mean_absolute_error(y_train, y_train_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print(f"Training MAE: {mae}")
print(f"Training RMSE: {rmse}")

# save
with open('data/model.pkl', 'wb') as _:
    pickle.dump(model, _)

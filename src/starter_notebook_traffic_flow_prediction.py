# Real-Time Traffic Prediction with Kafka - Starter Notebook
import os
# -----------------------------------------
# Kafka Installation Guide (with Zookeeper)
# -----------------------------------------

# Step 1: Download Kafka
# ----------------------
# 1. Go to the official Kafka website: https://kafka.apache.org/downloads
# 2. Download the latest stable version that supports Scala 2.13:
#    - Kafka Version: kafka_2.13-3.8.0 (or the latest stable version)
# 3. Unzip the downloaded file into your preferred installation directory:
#    Example:
#    tar -xzf kafka_2.13-3.8.0.tgz

# Step 2: Start Zookeeper
# -----------------------
# Kafka requires Zookeeper for cluster management. 
# You can start Zookeeper using the following command:
# Navigate to your Kafka installation directory and run:
#    bin/zookeeper-server-start.sh config/zookeeper.properties
# This starts Zookeeper on the default port (2181). Keep this terminal running.

# Step 3: Start Kafka Broker
# --------------------------
# After Zookeeper is up and running, start Kafka broker using:
#    bin/kafka-server-start.sh config/server.properties
# This starts the Kafka broker on the default port (9092). Keep this terminal running as well.

# Step 4: Create a Kafka Topic (for testing)
# ------------------------------------------
# Once Kafka is running, you can create a topic to send and consume data:
#    bin/kafka-topics.sh --create --topic traffic_data --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# Step 5: Check Topic
# -------------------
# You can check if the topic was created successfully by running:
#    bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
# The "traffic_data" topic should appear in the list.

# Step 6: Start Kafka Console Producer and Consumer (Optional for Testing)
# ------------------------------------------------------------------------
# You can use Kafka's console producer and consumer tools to manually send and receive messages for testing:
# 1. Console Producer:
#    bin/kafka-console-producer.sh --broker-list localhost:9092 --topic traffic_data
# 2. Console Consumer:
#    bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic traffic_data --from-beginning
# These tools are useful to verify that Kafka is working correctly before running your Python scripts.

# -----------------------------------------------
# Data Preparation, Producer and Consumer Scripts
# -----------------------------------------------

# Importing Necessary Libraries
import pickle

import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PyQt6.sip import array
from numpy import ndarray
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.sparse import csr_matrix, csc_matrix
import gzip
import json

from statsmodels.stats.descriptivestats import pd_ptp

from src.read_data import get_file_name

# Phase 1: Data Preparation

# Step 1: Load the Traffic Flow Forecasting Dataset from the .mat File
# ---------------------------------------------
# Guide for Creating Kafka Producer and Consumer
# ---------------------------------------------

# Phase 2: Kafka Producer and Consumer
# Guide:
# In this phase, you need to create two Python scripts: one for the producer and one for the consumer.
# The producer will read the data and send it to a Kafka topic, and the consumer will consume it in real-time.

# Create a producer script (producer.py) that sends the data in small chunks to Kafka.
# Create a consumer script (consumer.py) that listens to the Kafka topic and processes the messages.
# You can use the partial code snippets below as a guide.

# Producer Script Example (Partial)


# Consumer Script Example (Partial)
# Consume and process messages from Kafka topic
# You will implement this as shown in the consumer.py script

# ----------------------------------------
# Exploratory Data Analysis (EDA) Example
# ----------------------------------------

# Load the training data from the pickle files
with open(get_file_name('tra_X_tr'), 'rb') as f:
    tra_X_tr = pickle.load(f)

with open(get_file_name('tra_Y_tr'), 'rb') as f:
    tra_Y_tr = pickle.load(f)

with open(get_file_name('tra_adj_mat'), 'rb') as f:
    tra_adj_mat = pickle.load(f)

location_index = 0
traffic_flow_series = tra_Y_tr[location_index]
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

# Guide:
# In this phase, you should further explore the time-series nature of the data by generating more visualizations.
# You may want to explore different locations, investigate daily or weekly seasonality, or look at speed and occupancy data if available.

# ----------------------------------------
# Feature Engineering and Model Training
# ----------------------------------------

# Feature Engineering Example
# rolling_window_size = 4  # Rolling average over 1 hour (4 intervals)
# df_tra_X_tr['rolling_mean'] = df_tra_X_tr.iloc[:, 0].rolling(window=rolling_window_size).mean()
#
# # Fill any NaN values resulting from the rolling operation
# df_tra_X_tr.fillna(method='bfill', inplace=True)
#
# # Train a Simple Linear Regression Model
# X_train = df_tra_X_tr[['rolling_mean']]  # Example feature
# y_train = df_tra_Y_tr.values  # Assuming this is the target variable
#
# # Initialize and train the model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Evaluate the model
# y_train_pred = model.predict(X_train)
# mae = mean_absolute_error(y_train, y_train_pred)
# rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
# print(f"Training MAE: {mae}")
# print(f"Training RMSE: {rmse}")
#
# # Guide:
# # Add more features, try different models, and integrate Kafka producer and consumer scripts into your pipeline.

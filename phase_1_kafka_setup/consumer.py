import json
import pickle

import numpy as np
from kafka import KafkaConsumer
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.prediction import predict_traffic, get_model
from src.util import get_data_from_pickle


def subscribe(consumer_instance):
    model = get_model()
    predicted = [None] * 840
    try:
        for event in consumer_instance:
            key = event.key.decode("utf-8")
            value = event.value.decode("utf-8")
            value = json.loads(value)
            print(f"Message Received: ({key}, {value})")
            predicted[int(key)] = predict_traffic(value, model)
        consumer_instance.close()
    except Exception as ex:
        print('Exception in subscribing')
        print(str(ex))
    actual = get_data_from_pickle('tra_Y_te')[0]
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"Testing MAE: {mae}")
    print(f"Testing RMSE: {rmse}")
    _plot(predicted, actual)


def _plot(predicted, actual):
    plt.figure(figsize=(10, 5))
    plt.plot(predicted, label='Predicted')
    plt.plot(actual, label='Actual')
    plt.title(f"Traffic Flow predicted by model final")
    plt.xlabel("Time (15-minute intervals)")
    plt.ylabel("Traffic Flow")
    plt.savefig(f"Prediction with all features.jpg")
    plt.legend(loc='upper left')
    plt.show()


def get_kafka_consumer(topic_name, servers='localhost:9092'):
    _consumer = None
    try:
        _consumer = KafkaConsumer(topic_name, auto_offset_reset='earliest', bootstrap_servers=[servers],
                                  api_version=(0, 10), consumer_timeout_ms=10000)
    except Exception as ex:
        print('Exception while connecting Kafka')
        print(str(ex))
    finally:
        return _consumer


if __name__ == '__main__':
    consumer = get_kafka_consumer('datav3')
    subscribe(consumer)

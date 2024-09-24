import json
from time import sleep

from kafka import KafkaProducer
from scipy.sparse import csc_matrix

from src.util import get_data_from_pickle


def publish(producer_instance: KafkaProducer, topic_name, key, value: dict):
    try:
        key_bytes = bytes(key, encoding='utf-8')
        producer_instance.send(topic_name, key=key_bytes, value=value)
        producer_instance.flush()
        print(f"Publish Succesful ({key}, {value}) -> {topic_name}")
    except Exception as ex:
        print('Exception in publishing message')
        print(str(ex))


def get_kafka_producer(server='localhost:9092'):
    _producer = None
    try:
        _producer = KafkaProducer(bootstrap_servers=[server], api_version=(0, 10))
    except Exception as ex:
        print('Exception while connecting Kafka')
        print(str(ex))
    finally:
        return _producer


def stream_data():
    tra_x_te = get_data_from_pickle('tra_X_te')
    data = tra_x_te[0]
    location_index = 0
    producer = get_kafka_producer()
    for index in range(len(data)):
        timestamp_data = data.item(index).toarray()
        location_data = timestamp_data[location_index]
        message: csc_matrix = location_data
        features = {}
        for i in range(message.shape[0]):
            features[f'{i}'] = message[i]
        features = json.dumps(features).encode('utf-8')
        print(f'sending data at index {index} and location {location_index + 1}')
        publish(producer, 'datav3', key=f'{index}', value=features)  # Send message to Kafka topic
        print(f"Produced: {features}")
        sleep(1)


if __name__ == '__main__':
    stream_data()

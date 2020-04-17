#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# import numpy as np
# import pandas as pd
# import torch
# from torch.autograd import Variable
#
# from surprise import SVDpp, Dataset, Reader, NormalPredictor, accuracy, SVD
#
# from surprise.model_selection import cross_validate, train_test_split
#
# ratings = pd.read_csv('data/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'])[:1000]
# ratings = ratings[['user_id', 'movie_id', 'rating']]
# reader = Reader(line_format='user item rating', sep='::')
# data = Dataset.load_from_df(ratings, reader=reader)
# # data = cross_validate(NormalPredictor(), data, cv=2)
# train, test = train_test_split(data, test_size=0, train_size=1.0, shuffle=False)
# svd = SVD(n_factors=30, n_epochs=100)
# svd.fit(train)
# train, test = train_test_split(data, test_size=1.0, train_size=0, shuffle=False)
# X = svd.test(test)
# x = [i.r_ui for i in X]
# x = pd.DataFrame(x)
# x = pd.concat([ratings, x], axis=1)
# accuracy.mae(X)
# accuracy.rmse(X)

# import numpy as np
# import tensorflow as tf
# from tfcf.metrics import mae
# from tfcf.metrics import rmse
# from tfcf.datasets import ml1m
# from tfcf.config import Config
# from tfcf.models.svd import SVD
# from tfcf.models.svdpp import SVDPP
# from sklearn.model_selection import train_test_split
#
# x, y = ml1m.load_data()
#
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=0)
#
# config = Config()
# config.num_users = np.max(x[:, 0]) + 1
# config.num_items = np.max(x[:, 1]) + 1
# config.min_value = np.min(y)
# config.max_value = np.max(y)
#
# with tf.Session() as sess:
#     # For SVD++ algorithm, if `dual` is True, then the dual term of items'
#     # implicit feedback will be added into the original SVD++ algorithm.
#     # model = SVDPP(config, sess, dual=False)
#     # model = SVDPP(config, sess, dual=True)
#     model = SVD(config, sess)
#     model.train(x_train, y_train, validation_data=(
#         x_test, y_test), epochs=2, batch_size=3000)
#
#     y_pred = model.predict(x_test)
#     print(y_pred)
#     print('rmse: {}, mae: {}'.format(rmse(y_test, y_pred), mae(y_test, y_pred)))
#
#     # Save model
#     model = model.save_model('model/')


# from scipy.sparse.linalg import svds
#
# R = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0).as_matrix()
# user_ratings_mean = np.mean(R, axis=1)
# R_demeaned = R - user_ratings_mean.reshape(-1, 1)
# R_demeaned = R
#
# U, sigma, Vt = svds(R_demeaned, k=30, maxiter=200, tol=1e-10)
# sigma = np.diag(sigma)
# print(U.shape)
# print(sigma.shape)
# print(Vt.shape)

import time
import threading
import requests

# def get():
#     time.sleep(1)
#     for i in range(1000):
#         requests.get('http://127.0.0.1:8000/')
#         print(i)
#
#
# for i in range(100):
#     threading.Thread(target=get).start()

# from requests.auth import HTTPDigestAuth
#
# r = requests.get(url='http://es-cn-0pp16iw910008stel.public.elasticsearch.aliyuncs.com:9200/es_item_index_test', auth=('elastic', 'Zhuangdian!@#'))
# print(r)
# print(r.text)

# from pyalink.alink import *
# resetEnv()
# useLocalEnv(2)
# schema = "age bigint, workclass string, fnlwgt bigint, education string, education_num bigint, marital_status string, occupation string, relationship string, race string, sex string, capital_gain bigint, capital_loss bigint, hours_per_week bigint, native_country string, label string"
# adult_batch = CsvSourceStreamOp() \
#     .setFilePath("http://alink-dataset.cn-hangzhou.oss.aliyun-inc.com/csv/adult_train.csv") \
#     .setSchemaStr(schema)
# adult_batch.print()
# # sample = SampleStreamOp().setRatio(0.01).linkFrom(adult_batch)
# # sample.print(key="adult_data", refreshInterval=3)
# StreamOperator.execute()
import pandas as pd
#
# a = [{'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}]
# d = pd.DataFrame(a)
# d = d.append({'a': 0, 'c': 2}, ignore_index=True)
# print(d)

import json
import socket
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

bootstrap_servers = '172.16.100.31:9092,172.16.100.29:9092,172.16.100.30:9092'
producer = KafkaProducer(bootstrap_servers=bootstrap_servers.split(','),
                         api_version=(0, 10),
                         retries=5)
topic_name = 'Topic_Live_Heartbeat_Msg'

partitions = producer.partitions_for(topic_name)
d = pd.read_csv('live_data.csv').astype('int32').to_dict('records')
for i in d[:1]:
    future = producer.send(topic_name, value=json.dumps(i).encode('utf-8'), key='test'.encode('utf-8'))
    # producer.flush()
    future.get()

# consumer_id = 'Gid_Real_Time_Live_Heartbeat_Msg'
# consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers,
#                          group_id=consumer_id,
#                          api_version=(0, 10))

# class Kafka(object):
#     def __init__(self, broker):
#         self.broker = broker
#
#     def producer(self):
#         kafka_producer = KafkaProducer(bootstrap_servers=self.broker)
#         return kafka_producer
#
#     def consumer(self, topic):
#         kcm = KafkaConsumer(topic, bootstrap_servers=self.broker)
#         return kcm



# bootstrap_servers = '172.16.100.31:9092,172.16.100.29:9092,172.16.100.30:9092'
# topic_name = 'Topic_Live_Heartbeat_Msg'
# consumer_id = 'Gid_Real_Time_Live_Heartbeat_Msg'
# data = KafkaSourceStreamOp() \
#     .setBootstrapServers(bootstrap_servers) \
#     .setTopic(topic_name) \
#     .setStartupMode("LATEST") \
#     .setGroupId(consumer_id)
# col = ['source', 'shop_authentication_type', 'shop_level', 'shop_cid_1', 'type', 'room_type', 'gender', 'user_level', 'live_if_new', 'user_player_level',
#        'shop_fans_count', 'auction_count_1', 'label']
# data = JsonValueStreamOp().setJsonPath(["$." + i for i in col]).setSelectedCol("message").setOutputCols(col).linkFrom(data).select(col)
# data.print()
# StreamOperator.execute()

import tensorflow as tf
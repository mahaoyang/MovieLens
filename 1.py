#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

from surprise import SVDpp, Dataset, Reader, NormalPredictor, accuracy, SVD

from surprise.model_selection import cross_validate, train_test_split

ratings = pd.read_csv('data/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'])[:1000]
ratings = ratings[['user_id', 'movie_id', 'rating']]
reader = Reader(line_format='user item rating', sep='::')
data = Dataset.load_from_df(ratings, reader=reader)
# data = cross_validate(NormalPredictor(), data, cv=2)
train, test = train_test_split(data, test_size=0, train_size=1.0, shuffle=False)
svd = SVD(n_factors=30, n_epochs=100)
svd.fit(train)
train, test = train_test_split(data, test_size=1.0, train_size=0, shuffle=False)
X = svd.test(test)
x = [i.r_ui for i in X]
x = pd.DataFrame(x)
x = pd.concat([ratings, x], axis=1)
accuracy.mae(X)
accuracy.rmse(X)

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

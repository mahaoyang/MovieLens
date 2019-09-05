#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import os
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegressionCV
from surprise import SVDpp, Dataset, Reader, NormalPredictor, accuracy, SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split as surprise_train_test_split

from lgb_util import *

print('loading data...')
print(datetime.now())
movies = pd.read_csv('data/movies.dat', sep='::', names=['movie_id', 'title', 'genres'])
ratings = pd.read_csv('data/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'])
users = pd.read_csv('data/users.dat', sep='::', names=['user_id', 'gender', 'age', 'occupation', 'Zip-code'])

print('feature processing...')
genres = set()
for i in movies['genres'].values.tolist():
    [genres.add(ii) for ii in i.strip().split('|')]
genres_length = len(genres)
genres = dict(zip(list(genres), [i for i in range(len(genres))]))
movies_genres = pd.DataFrame(movies['genres'].map(lambda x: trans_genres(x, genres_length, genres)).values.tolist())
movies_genres.columns = list(genres.keys())
movies['publish_years'] = movies['title'].map(lambda x: trans_publish_years(x))
movies = pd.concat([movies, movies_genres], axis=1, ignore_index=False).drop(columns=['genres'])
users['age'] = users['age'].map(lambda x: 0 if x <= 6 else x)
ratings = ratings[['user_id', 'movie_id', 'rating']]
# ratings['rating'] = ratings['rating'].map(lambda x: 0 if x < 4 else 1)
if not os.path.exists('feature/raw_svd_fi.pkl'):
    reader = Reader()
    data = Dataset.load_from_df(ratings, reader=reader)
    train, test = surprise_train_test_split(data, test_size=0, train_size=1.0, shuffle=False)
    svd = SVD(n_factors=500, n_epochs=100, random_state=321)
    svd.fit(train)
    svd_fu = pd.concat([ratings['user_id'].drop_duplicates().reset_index(drop=True), pd.DataFrame(svd.pu.tolist())],
                       axis=1)
    svd_fi = pd.concat([ratings['movie_id'].drop_duplicates().reset_index(drop=True), pd.DataFrame(svd.qi.tolist())],
                       axis=1)
    train, test = surprise_train_test_split(data, test_size=1.0, train_size=0, shuffle=False)
    svd_pred = svd.test(test)
    svd_pred = pd.DataFrame([i.r_ui for i in svd_pred])
    with open('feature/raw_svd_fu.pkl', 'wb') as f:
        pickle.dump(svd_fu, f)
    with open('feature/raw_svd_fi.pkl', 'wb') as f:
        pickle.dump(svd_fi, f)
    with open('feature/raw_svd_pred.pkl', 'wb') as f:
        pickle.dump(svd_pred, f)
else:
    with open('feature/raw_svd_fu.pkl', 'rb') as f:
        svd_fu = pickle.load(f)
    with open('feature/raw_svd_fi.pkl', 'rb') as f:
        svd_fi = pickle.load(f)
    with open('feature/raw_svd_pred.pkl', 'rb') as f:
        svd_pred = pickle.load(f)
# ratings['rating'] = ratings['rating'].map(lambda x: 0 if x < 4 else 1)
ratings = pd.merge(ratings, users, how='left', on='user_id')
ratings = pd.merge(ratings, svd_fu, how='left', on='user_id')
ratings = pd.merge(ratings, movies, how='left', on='movie_id')
ratings = pd.merge(ratings, svd_fi, how='left', on='movie_id')
ratings = pd.concat([ratings, svd_pred], axis=1)
for i in ratings.columns:
    ratings[i] = LabelEncoder().fit_transform(ratings[i])
print(datetime.now())

print('data processing...')
X = ratings.drop(columns='rating')
# x = OneHotEncoder().fit_transform(x)
Y = ratings['rating']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=321)

print('start training...')
lgb_train = lgb.Dataset(x_train, y_train)
lgb_test = lgb.Dataset(x_test, y_test)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型

    # 'objective': 'binary',  # 目标函数
    # 'metric': {'binary_logloss'},  # 评估函数

    # 'objective': 'multiclass',  # 目标函数
    # 'num_class': 5,
    # 'metric': {'multi_logloss'},  # 评估函数

    'objective': 'regression',  # 目标函数
    'metric': {'l2'},  # 评估函数

    'scale_pos_weight ': 1000,
    'max_delta_step ': 0.9,
    'num_leaves': 500,  # 叶子节点数
    'max_depth': 20,
    # 'max_bin': 100,
    'learning_rate': 0.1,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 10,  # k 意味着每 k 次迭代执行bagging
    'top_k': 30,
    'verbose': -1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}
gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=lgb_test, early_stopping_rounds=10)
gbm.save_model('lgb_model.txt')

print('lgb predicting...')
lgb_pred = gbm.predict(x_train, num_iteration=gbm.best_iteration, pred_leaf=False)
# binary classify & regression reshape
lgb_pred = lgb_pred.reshape((-1, 1))

print('lr training...')
lr_cv = LogisticRegressionCV(Cs=10, cv='warn', penalty='l2', tol=1e-4, max_iter=10, n_jobs=1, random_state=321)
lr_cv.fit(lgb_pred.tolist(), y_train.values.tolist())

print('start predicting...')
lgb_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration, pred_leaf=False)
# binary classify & regression reshape
lgb_pred = lgb_pred.reshape((-1, 1))
y_pred = lr_cv.predict(lgb_pred.tolist())

print(y_pred.tolist())
print('MAE is ', mean_absolute_error(y_test.values.tolist(), y_pred.tolist()))
print('RMSE is ', np.sqrt(mean_squared_error(y_test.values.tolist(), y_pred.tolist())))
print('accuracy is ', accuracy_score(y_test.values.tolist(), y_pred.tolist()))
# print('f1 score is ', f1_score(y_test.values.tolist(), y_pred.tolist()))

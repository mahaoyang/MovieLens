#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegressionCV
from keras import losses
from deepctr.models import *
from deepctr.inputs import SparseFeat, get_fixlen_feature_names, DenseFeat

from lgb_util import *

print('loading data...')
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
ratings = ratings[['user_id', 'movie_id', 'rating']]
ratings['rating'] = ratings['rating'].map(lambda x: 0 if x < 4 else 1)
ratings = pd.merge(ratings, users, how='left', on='user_id')
ratings = pd.merge(ratings, movies, how='left', on='movie_id').fillna(0)
for i in ratings.columns:
    ratings[i] = LabelEncoder().fit_transform(ratings[i])
print('data processing...')
x = ratings.drop(columns='rating')
y = ratings['rating']
x.columns = [str(i) for i in range(len(x.columns))]
# sparse_features = [str(i) for i in range(8, len(x.columns))]
# dense_features = [str(i) for i in range(8)]
# for i in x.columns:
#     x[i] = LabelEncoder().fit_transform(x[i])
# mms = MinMaxScaler(feature_range=(0, 1))
# x[dense_features] = mms.fit_transform(x[dense_features].astype('int32'))
# fixlen_feature_columns = [SparseFeat(feat, x[feat].nunique())
#                           for feat in sparse_features] + [DenseFeat(feat, 1, )
#                                                           for feat in dense_features]
fixlen_feature_columns = [SparseFeat(feat, x[feat].nunique())
                          for feat in x.columns]
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=321)

print('start training...')
# lr_cv = LogisticRegressionCV(Cs=10, cv=10, penalty='l2', tol=1e-4, max_iter=10, n_jobs=1, random_state=321)
# lr_cv.fit(lgb_pred.tolist(), y_train.values.tolist())

# x_train = x_train.reset_index(drop=True)
# y_train = y_train.reset_index(drop=True)

train_model_input = [x_train[name] for name in fixlen_feature_names]
model = PNN(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", loss=losses.mae, metrics=['accuracy', 'mse'], )
history = model.fit(train_model_input, y_train.values,
                    batch_size=20480, epochs=10, verbose=2, validation_split=0.2, )

deep_pred = model.predict(train_model_input, batch_size=20480)
lr_cv = LogisticRegressionCV(Cs=10, cv='warn', penalty='l2', tol=1e-4, max_iter=10, n_jobs=1, random_state=321)
lr_cv.fit(deep_pred.tolist(), y_train.values.tolist())

print('start predicting...')
# y_pred = lr_cv.predict(lgb_pred.tolist())

# x_test = x_test.reset_index(drop=True)
# y_test = y_test.reset_index(drop=True)
x_test.columns = [str(i) for i in range(len(x_test.columns))]
feat = [x_test[name] for name in fixlen_feature_names]
deep_pred = model.predict(feat, batch_size=10240)
y_pred = lr_cv.predict(deep_pred.tolist())

print('The final accuracy is ', accuracy_score(y_test.values.tolist(), y_pred.tolist()))
print('The final f1 score is ', f1_score(y_test.values.tolist(), y_pred.tolist()))

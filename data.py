#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
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
count = ratings[ratings['age'] == 1]['age'].value_counts()
# count = [ratings[i].value_counts() for i in ratings.columns]
# for i in count:#     i.plot.bar(figsize=(6, 6))
#     plt.show()
print()

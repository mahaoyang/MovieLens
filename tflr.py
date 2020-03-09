#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegressionCV
from keras import losses
from keras import layers
from keras import models
from keras.utils import to_categorical
from deepctr.models import *
from deepctr.inputs import SparseFeat, get_feature_names, DenseFeat

from lgb_util import *

print('loading data...')
movies = pd.read_csv('data/movies.dat', sep='::', names=['movie_id', 'title', 'genres'])
ratings = pd.read_csv('data/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'])
users = pd.read_csv('data/users.dat', sep='::', names=['user_id', 'gender', 'age', 'occupation', 'Zip-code'])
ratings = pd.merge(ratings, users, how='left', on='user_id')
ratings = pd.merge(ratings, movies, how='left', on='movie_id').fillna(0)

# This returns a tensor
inputs = layers.Input(shape=(3,))

# a layer instance is callable on a tensor, and returns a tensor
output_1 = layers.Dense(2, activation='sigmoid')(inputs)
output_2 = layers.Dense(2, activation='sigmoid')(output_1)
output_3 = layers.Dense(2, activation='sigmoid')(output_2)
cat = layers.concatenate([output_1, output_2, output_3])
predictions = layers.Dense(2, activation='softmax')(cat)

# This creates a model that includes
# the Input layer and three Dense layers
model = models.Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
data = ratings[['movie_id', 'user_id', 'rating']][-200:].values
labels = ratings['rating'].map(lambda x: 1 if int(x) > 2 else 0)[-200:].values
model.fit(data, to_categorical(labels), epochs=10, batch_size=500)  # starts training
print(model.predict(data[-10:]))

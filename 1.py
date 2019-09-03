#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import pandas as pd

from surprise import SVDpp, Dataset, Reader, NormalPredictor, accuracy
from surprise.model_selection import cross_validate, train_test_split

ratings = pd.read_csv('data/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'])[:1000]
ratings = ratings[['user_id', 'movie_id', 'rating']]
reader = Reader(line_format='user item rating', sep='::')
data = Dataset.load_from_df(ratings, reader=reader)
# data = cross_validate(NormalPredictor(), data, cv=2)
train, test = train_test_split(data, test_size=0, train_size=1.0, shuffle=False)
svd = SVDpp(n_factors=30, n_epochs=300)
svd.fit(train)
train, test = train_test_split(data, test_size=1.0, train_size=0, shuffle=False)
X = svd.test(test)
accuracy.mae(X)
accuracy.rmse(X)

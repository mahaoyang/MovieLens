#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import pandas as pd
from surprise import SVDpp, Dataset, Reader, NormalPredictor
from surprise.model_selection import cross_validate

ratings = pd.read_csv('data/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'])
ratings = ratings[['user_id', 'movie_id', 'rating']]
reader = Reader(line_format='user item rating', sep='::')
data = Dataset.load_from_df(ratings, reader=reader)
data = cross_validate(NormalPredictor(), data, cv=2)

svd = SVDpp(n_factors=2, n_epochs=2)
X = svd.fit(data)
print(X)
print(X.shape)

#!/usr/bin/python3
# -*- encoding: utf-8 -*-
from surprise import SVDpp
from sklearn.random_projection import sparse_random_matrix

X = sparse_random_matrix(10000, 10000, density=0.01, random_state=42)
print(X)
svd = SVDpp(n_components=2, n_iter=7, random_state=42)
X = svd.fit_transform(X)
print(X)
print(X.shape)
print(svd.singular_values_)

#!/usr/bin/python3
# -*- encoding: utf-8 -*-
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix

X = sparse_random_matrix(100, 10, density=0.01, random_state=42)
print(X)
svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
X = svd.fit_transform(X)
print(X)
print(svd.singular_values_)

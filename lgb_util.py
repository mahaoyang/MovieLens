#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import re
import math


def trans_publish_years(s):
    s = re.findall('[\d+]', s)
    return ''.join(s)[-4:]


def trans_genres(genre, genres_length, genres):
    ger = [0] * genres_length
    for g in genre.strip().split('|'):
        ger[genres[g]] = 1
    return ger


def sigmoid(x):
    1 / (1 + math.exp(-x))


if __name__ == '__main__':
    print(trans_publish_years('drrg 4 hfgh 2646'))

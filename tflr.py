#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
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
inputs = layers.Input(shape=(3,), name='input_3')

# a layer instance is callable on a tensor, and returns a tensor
output_1 = layers.Dense(2, activation='sigmoid', name='lr')(inputs)
# output_2 = layers.Dense(2, activation='sigmoid')(output_1)
# output_3 = layers.Dense(2, activation='sigmoid')(output_2)
# cat = layers.concatenate([output_1, output_2, output_3])
# predictions = layers.Dense(2, activation='softmax')(cat)
# predictions = layers.Dense(2, activation='softmax', name='predict')(output_1)

# This creates a model that includes
# the Input layer and three Dense layers
model = models.Model(inputs=inputs, outputs=output_1)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
data = ratings[['movie_id', 'user_id', 'rating']][-200:].values
labels = ratings['rating'].map(lambda x: 1 if int(x) > 2 else 0)[-200:].values
model.fit(data, to_categorical(labels), epochs=10, batch_size=500)  # starts training
print(data[-10:].tolist())
print(model.predict(data[-10:]).tolist())


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


# save keras model as tf pb files ===============
from keras import backend as K

wkdir = 'C:/Users/99263/PycharmProjects/MovieLens'
pb_filename = 'lr.pb'
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)


def get_all_layernames():
    import os
    """get all layers name"""
    pb_file_path = os.path.join('C:/Users/99263/PycharmProjects/MovieLens', 'lr.pb')

    from tensorflow.python.platform import gfile

    sess = tf.Session()
    # with gfile.FastGFile(pb_file_path + 'model.pb', 'rb') as f:
    with gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        for tensor_name in tensor_name_list:
            print(tensor_name, '\n')
    return


get_all_layernames()
file = "C:/Users/99263/PycharmProjects/MovieLens/lr.pb"
with tf.gfile.FastGFile(file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    ele = ["input_3:0", "lr/Sigmoid:0"]
    # ele = ["prediction_layer/Reshape:0", "age:0", "room_type:0", "source:0", "user_player_level_score:0"]
    age, result = tf.import_graph_def(
        graph_def, return_elements=ele)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    result = sess.run(result, feed_dict={age: [[0.5, 2, 3], [0.5, 2, 3]]})
    print(result)

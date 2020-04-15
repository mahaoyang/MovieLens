#!/usr/bin/python3
# -*- encoding: utf-8 -*-

from deepctr import models
import tensorflow as tf
import numpy as np

# with tf.Session() as sess:
#     with tf.device("/gpu:0"):
#         matrix1 = tf.constant([[3., 3.]])
#         matrix2 = tf.constant([[2.], [2.]])
#         product = tf.matmul(matrix1, matrix2)
#         result = sess.run([matrix1, matrix1, product])
#         print(result)

# sess = tf.InteractiveSession()
#
# x = tf.Variable([1.0, 2.0])
# a = tf.constant([3.0, 3.0])
#
# # 使用初始化器 initializer op 的 run() 方法初始化 'x'
# # print(x.initializer.run())
# x.initializer.run()
# print(x.eval())
#
# # 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果
# sub = tf.subtract(x, a)
# print(sub.eval())

# # 创建一个变量, 初始化为标量 0.
# state = tf.Variable(0, name="counter")
#
# # 创建一个 op, 其作用是使 state 增加 1
#
# one = tf.constant(1)
# new_value = tf.add(state, one)
# update = tf.assign(state, new_value)
#
# # 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# # 首先必须增加一个`初始化` op 到图中.
# init_op = tf.initialize_all_variables()
#
# # 启动图, 运行 op
# with tf.Session() as sess:
#     # 运行 'init' op
#     sess.run(init_op)
#     # 打印 'state' 的初始值
#     print(sess.run(state))
#     # 运行 op, 更新 'state', 并打印 'state'
#     for _ in range(3):
#         sess.run(update)
#         print(sess.run(state))


# print(tf.add(1, 2))
# print(tf.add([1, 2], [3, 4]))
# print(tf.square(5))
# print(tf.reduce_sum([1, 2, 3]))
#
# # Operator overloading is also supported
# print(tf.square(2) + tf.square(3))

# x = tf.matmul([[1]], [[2, 3]])
# print(x)
# print(x.shape)
# print(x.dtype)

# import random
#
# print(random.uniform(0.99, 1))


# import numpy as np
# import tensorflow as tf
#
# # 定义特性列，线性模型中特性是列是x，shape=[1]，因此定义如下：
# feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
#
# # 使用tf.estimator内置的LinearRegressor来完成线性回归算法
# # tf.estimator提供了很多常规的算法模型以便用户调用，不需要用户自己重复造轮子
# # 到底为止，短短两行代码我们的建模工作就已经完成了
# estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
#
# # 有了模型之后，我们要使用模型完成训练->评估->预测这几个步骤
# # 训练数据依旧是(1.,0.)，(2.,-1.)，(3.,-2.)，(4.,-3.)这几个点，拆成x和y两个维度的数组
# x_train = np.array([1., 2., 3., 4.])
# y_train = np.array([0., -1., -2., -3.])
#
# # 评估数据为(2.,-1.01)，(5.,-4.1)，(8.,-7.)，(1.,0.)这四个点，同样拆分成x和y两个维度的数组
# x_eval = np.array([2., 5., 8., 1.])
# y_eval = np.array([-1.01, -4.1, -7., 0.])
#
# # 用tf.estimator.numpy_input_fn方法生成随机打乱的数据组，每组包含4个数据
# input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
# # 循环1000次训练模型
# estimator.train(input_fn=input_fn, steps=1000)
#
# # 生成训练数据，分成1000组，每组4个数据
# train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
# # 生成评估数据，分成1000组，每组4个数据
# eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)
#
# # 训练数据在模型上的预测准确率
# train_metrics = estimator.evaluate(input_fn=train_input_fn)
# # 评估数据在模型上的预测准确率
# eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
#
# print("train metrics: %r" % train_metrics)
# print("eval metrics: %r" % eval_metrics)
# def raw_serving_input_fn():
#     serialized_tf_example = tf.placeholder(tf.float32, shape=[1], name="images")
#     features = {"images": serialized_tf_example}
#     receiver_tensors = {'predictor_inputs': serialized_tf_example}
#     return tf.estimator.export.build_raw_serving_input_receiver_fn(features, receiver_tensors)
# feature_spec = {"x": tf.placeholder(dtype=tf.float32, shape=[1])}
# serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
# estimator.export_savedmodel('C:/Users/99263/PycharmProjects/MovieLens', serving_input_fn)

def get_all_layernames():
    import os
    """get all layers name"""
    pb_file_path = os.path.join('C:/Users/99263/PycharmProjects/MovieLens', 'model.pb')

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
file = "C:/Users/99263/PycharmProjects/MovieLens/model.pb"
with tf.gfile.FastGFile(file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    ele = ["prediction_layer/Reshape:0", "age:0", "room_type:0", "source:0", "user_player_level_score:0"]
    # ele = ["prediction_layer/Reshape:0", "age:0", "room_type:0", "source:0", "user_player_level_score:0"]
    result, age, room_type, source, user_player_level_score = tf.import_graph_def(
        graph_def, return_elements=ele)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    result = sess.run(result, feed_dict={age: [[0.5], [0.5]], room_type: [[4.0], [2]], source: [[0.5], [0.7]], user_player_level_score: [[1.0], [2]]})
    print(result)

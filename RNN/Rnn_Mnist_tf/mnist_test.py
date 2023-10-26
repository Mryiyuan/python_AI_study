import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 加载mnist数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data(path='D:\code\python_AI_study\Rnn_Mnist\data\mnist.npz')
# print(x_train.shape)
# plt.imshow(x_train[0])
x_train, x_test = x_train / 255.0, x_test / 255.0  # 归一化
y_train_onehot = tf.keras.utils.to_categorical(y_train)  # 将标签转换成独热编码
y_test_onehot = tf.keras.utils.to_categorical(y_test)

network = tf.keras.models.load_model('model.h5')
network.evaluate(x_test, y_test_onehot)

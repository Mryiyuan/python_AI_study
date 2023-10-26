import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 加载mnist数据集
mnist = tf.keras.datasets.mnist
path = 'D:\code\PycharmProjects\Rnn_Mnist\data\mnist.npz'#记得修改路径，建议新手先使用绝对路径
(x_train, y_train), (x_test, y_test) = mnist.load_data(path)
# print(x_train.shape)
# plt.imshow(x_train[0])
x_train, x_test = x_train / 255.0, x_test / 255.0  # 归一化
y_train_onehot = tf.keras.utils.to_categorical(y_train)  # 将标签转换成独热编码
y_test_onehot = tf.keras.utils.to_categorical(y_test)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 28*28
model.add(tf.keras.layers.Dense(128, activation='relu'))  # 中间隐藏层激活函数用relu
model.add(tf.keras.layers.Dense(10, activation='softmax'))  # 多分类输出一般用softmax分类器

# loss函数使用交叉熵
# 顺序编码用sparse_categorical_crossentropy
# 独热编码用categorical_crossentropy
lr = 0.01
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss='categorical_crossentropy',
              metrics=['acc'])

bz = 32
history = model.fit(x_train, y_train_onehot, epochs=5, batch_size=bz)

model.save('model.h5')
print('saved total model.')

# plt.plot(history.epoch, history.history.get('accuracy'), label='accuracy')
# plt.legend()
# plt.show()

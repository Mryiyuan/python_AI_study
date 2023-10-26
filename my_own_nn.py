import numpy as np
from sklearn.datasets import fetch_

# python写一个神经网络实现手写数字识别
# 激活函数是relu, 输出层是softmax分类

def train_y(y_true):
    y_ohe = np.zeros(10)
    y_ohe[int(y_true)]=1
    return y_ohe
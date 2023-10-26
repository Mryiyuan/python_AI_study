from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 这行代码使用 train_test_split 函数将数据集 X 和对应的标签 y 划分为训练集和测试集，按照指定的比例进行划分。具体来说：
#
# X 是特征矩阵，包含了所有的样本特征。
# y 是标签向量，包含了所有的样本标签。
# test_size=0.2 表示将数据集划分为训练集和测试集时，测试集所占的比例为 20%。
# random_state=42 是随机种子，用于控制数据集的划分方式，确保每次划分的结果是一致的。
# 划分完成后，返回了四个结果：
#
# X_train 是训练集的特征矩阵。
# 在特征矩阵中，每个元素表示一个样本的某个特征的取值。例如，在一个房屋价格预测的任务中，特征矩阵可能包含了房屋的各种特征，比如房屋的面积、卧室数量、浴室数量等。每一行代表一个房屋样本，每一列代表一种特征，特征矩阵中的每个元素就是对应房屋样本在该特征上的取值。
# 特征矩阵的大小通常为 m 行 n 列，其中 m 表示样本的数量，n 表示特征的数量。例如，如果有 100 个房屋样本，每个样本包含了 3 个特征，那么特征矩阵的大小就是 100 行 3 列。
# X_test 是测试集的特征矩阵。
# y_train 是训练集的标签向量。
# y_test 是测试集的标签向量。

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建多层感知器分类器
clf = MLPClassifier(hidden_layer_sizes=(100,50,25,5),
                    activation='relu',
                    solver='adam',
                    max_iter=1000,
                    random_state=42,
                    alpha=1e-5,
                    tol=1e-4,
                    verbose=True)

# hidden_layer_sizes=(100, 50, 25, 5): 这个参数指定了隐藏层的结构，具体来说，有 4 个隐藏层，分别包含 100、50、25 和 5 个神经元。
#
# activation='relu': 指定了激活函数为 ReLU（线性整流函数），这是一种常用的激活函数，有助于网络学习非线性关系。
#
# solver='adam': 指定了优化器为 Adam，Adam 是一种常用的随机梯度下降优化算法，通常在训练神经网络时表现良好。
#
# max_iter=1000: 指定了最大迭代次数，即训练过程中神经网络的最大迭代次数。
#
# random_state=42: 指定了随机种子，用于保证代码的可重复性，不同的种子会产生不同的随机初始值。
#
# alpha=1e-5: 正则化参数，控制模型的复杂度，避免过拟合。
#
# tol=1e-4: 迭代停止的容差值，当连续两次迭代之间的损失值差小于此值时，训练将停止。
#
# verbose=True: 设置为 True 时，会输出训练过程中的详细信息。
clf.fit(X_train, y_train)

# 在测试集上进行预测
# X_test 是测试集的特征矩阵。
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 手动输入数据
manual_input = [[5.1, 3.5, 1.4, 0.2]]  # 手动输入一组数据，例如鸢尾花的花萼长度、花萼宽度、花瓣长度、花瓣宽度

# 对输入数据进行预处理
manual_input_scaled = scaler.transform(manual_input)

# 使用模型进行预测
prediction = clf.predict(manual_input_scaled)

print("Predicted class:", prediction)

#mse + l1 = lasso
#mse + l2 = Ridge
#mse + l1 + l2 = ElasticNet
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

# 引用Ridge类写脊/岭回归
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
ridge_reg = Ridge(alpha=0.6, solver='sag')
ridge_reg.fit(X, y)
# print("X: ", X)
# print("y: ", y)
print("ridge_reg.predict(1.5): ", ridge_reg.predict([[1.5]]))
print("ridge_reg.intercept_: ", ridge_reg.intercept_)
print("ridge_reg.coef_: ", ridge_reg.coef_)


# ！引用Ridge类写脊/岭回归
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

sgd_reg = SGDRegressor(penalty='l2', max_iter=1000)
sgd_reg.fit(X, y.reshape(-1,))

print("sgd_reg.predict(1.5): ", sgd_reg.predict([[1.5]]))
print("sgd_reg.intercept_: ", sgd_reg.intercept_)
print("sgd_reg.coef_: ", sgd_reg.coef_)

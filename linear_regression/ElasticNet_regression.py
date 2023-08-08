
#mse + l1 = lasso
#mse + l2 = Ridge
#mse + l1 + l2 = ElasticNet
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor

# 引用Ridge类写脊/岭回归
X = 2 * np.random.randn(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
elasticNet_reg = ElasticNet(alpha=0.4, l1_ratio=0.15)
elasticNet_reg.fit(X, y)
# print("X: ", X)
# print("y: ", y)
print("elasticNet_reg.predict(1.5): ", elasticNet_reg.predict([[1.5]]))
print("elasticNet_reg.intercept_: ", elasticNet_reg.intercept_)
print("elasticNet_reg.coef_: ", elasticNet_reg.coef_)



sgd_reg = SGDRegressor(penalty='elasticnet', max_iter=1000)
sgd_reg.fit(X, y.reshape(-1,))

print("sgd_reg.predict(1.5): ", sgd_reg.predict([[1.5]]))
print("sgd_reg.intercept_: ", sgd_reg.intercept_)
print("sgd_reg.coef_: ", sgd_reg.coef_)

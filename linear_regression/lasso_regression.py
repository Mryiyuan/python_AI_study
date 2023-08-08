#mse + l1 = lasso
#mse + l2 = Ridge
#mse + l1 + l2 = ElasticNet
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

lasso_reg = Lasso(alpha=0.15, max_iter=30000)
lasso_reg.fit(X, y)

print("lasso_reg.predict(1.5): ", lasso_reg.predict([[1.5]]))
print("lasso_reg.intercept_: ", lasso_reg.intercept_)
print("lasso_reg.coef_: ", lasso_reg.coef_)
#---------------------------引用sgd-------------------------------
sgd_reg = SGDRegressor(penalty='l1', max_iter=10000)
sgd_reg.fit(X, y.reshape(-1,))

print("sgd_reg.predict(1.5): ", sgd_reg.predict([[1.5]]))
print("sgd_reg.intercept_: ", sgd_reg.intercept_)
print("sgd_reg.coef_: ", sgd_reg.coef_)
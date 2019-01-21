import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(99)

# q1
n = 1000
est_1 = 1
est_2 = np.log(2)
exp_rand = expon.rvs(np.ones(n))
mse_1 = ((exp_rand - est_1)**2).mean()
mse_2 = ((exp_rand - est_2)**2).mean()
mae_1 = np.abs(exp_rand - est_1).mean()
mae_2 = np.abs(exp_rand - est_2).mean()

# q2
a, b = 1.1, 2
x = np.linspace(-2, 2, 1000)
y = b * (np.exp(a * x) - a * x - 1)
plt.plot(x, y)
plt.title('Asymmetric Loss Function w.r.t z')
plt.xlabel('z')
plt.ylabel('Loss')

# q3
def ols_mse_vary_dim(p, n=100):
    X = np.random.randn(n, p)
    y = 4 * X[:, 0] + np.random.randn(n)
    test_X = np.random.randn(n, p)
    test_y = 4 * test_X[:, 0] + np.random.randn(n)
    reg = LinearRegression().fit(X, y)
    predict_y = reg.predict(test_X)
    return mean_squared_error(test_y, predict_y)


p_arr = np.arange(2, 81)
vec_fun = np.vectorize(ols_mse_vary_dim)
mse_arr = vec_fun(p_arr)
plt.plot(p_arr, mse_arr)
plt.title('OLS Erros in High-dim Space')
plt.xlabel('Number of Dimensions')
plt.ylabel('MSE')

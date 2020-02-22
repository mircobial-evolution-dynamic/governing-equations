import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

def RSM(t, pop):
    a_r, a_s, a_m, K, e, f = 1.1, 2.0, 1.1, 10**6, 0.00001, 0.00001
    R, S, M = pop
    return [a_r*R*(1-(R+S+M)/K),
            a_s*S*(1-(R+S+M)/K) - e*S*R - f*S*M,
            a_m*M*(1-(R+S+M)/K) + e*S*R + f*S*M]

tspan = np.linspace(0, 20, num=201)
ini = [10, 10000, 0]
sol = integrate.solve_ivp(RSM, [tspan[0], tspan[-1]], ini, method='RK45', t_eval=tspan)

# plt.plot(sol.t, sol.y[0].T, label='R')
# plt.plot(sol.t, sol.y[1].T, label='S')
# plt.plot(sol.t, sol.y[2].T, label='M')
# plt.legend()
# plt.show()
# print(sol.y.shape)

def data_aug(sol_):
    n = len(sol_)
    for i in range(n):
        sol_ = np.vstack((sol_, sol_[i]**2))
    for i in range(n):
        for j in range(i+1, n):
            sol_ = np.vstack((sol_, sol_[i]*sol_[j]))
    for i in range(n):
        for j in range(n, 2*n):
            sol_ = np.vstack((sol_, sol_[i]*sol_[j]))
    return sol_

p1 = data_aug(sol.y)
exp_data = np.diff(sol.y, axis=1)
p1 = p1[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(p1.T, exp_data.T, test_size=0.34, random_state= 10)
las = Lasso()
las.fit(X_train, y_test)
y_est = las.predict(X_test)

print(las.coef_)
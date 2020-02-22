import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import csv
from scipy import optimize
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

def RSM(t, pop):
    a_r, a_s, a_m, K, e, f = 1.1, 2.0, 1.1, 10**6, 0.00001, 0.00001
    R, S, M = pop
    return [a_r*R*(1-(R+S+M)/K),
            a_s*S*(1-(R+S+M)/K) - e*S*R - f*S*M,
            a_m*M*(1-(R+S+M)/K) + e*S*R + f*S*M]

tspan = np.linspace(0, 20, num=2001)
dt = 0.01
ini = [10, 10000, 0]
sol = integrate.solve_ivp(RSM, [tspan[0], tspan[-1]], ini, method='RK45', t_eval=tspan)

plt.plot(sol.t, sol.y[0].T, label='R')
plt.plot(sol.t, sol.y[1].T, label='S')
plt.plot(sol.t, sol.y[2].T, label='M')
plt.legend()
plt.show()
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
exp_data = np.diff(sol.y, axis=1) /dt
p1 = p1[:, 1:]

coeff_ = np.dot(np.linalg.pinv(p1.T), exp_data.T)
est = np.dot(p1.T, coeff_)
plt.plot(est[:,0], '-*', lw=4, color='salmon', label ='est_dR')
plt.plot(exp_data[0,:].T, color='red', label ='true_dR')
plt.plot(est[:,1],'-*', lw=4, color='skyblue', label ='est_dS')
plt.plot(exp_data[1,:].T, color='blue', label = 'true_dS')
plt.plot(est[:,2], '-*',lw=4, color='grey', label = 'est_dM')
plt.plot(exp_data[2,:].T, color='black', label = 'true_dM')
plt.legend()
plt.show()
print(est)

# csvfile = "./output.csv"
# with open(csvfile, "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     writer.writerows(coeff_)

# print(exp_data.shape)
# print(p1.shape)

# def fun(coeff_, p1, exp_data):
#     return np.dot(p1.T, coeff_) - exp_data.T
#
# coeff_guess = np.array([np.random.rand(18)]).T
# print(coeff_guess.shape)
# res_log = optimize.least_squares(fun, coeff_guess, loss='cauchy', f_scale=0.1,
#                                  args=(p1, exp_data))
# # coeff_set = []
# # for i in range(len(exp_data)):
# #     X_train, X_test, y_train, y_test = train_test_split(p1.T, np.expand_dims(exp_data[i,:], axis=0).T, test_size=0.1)
# #     las = Lasso(alpha=1, max_iter=10**5, tol=0.001)
# #     las.fit(X_train, y_train)
# #     y_est = las.predict(X_test)
# #     coeff_set.append(las.coef_)
#
#
# # for i in range(len(coeff_set)):
# #     est = np.dot(p1.T, coeff_set[i].T)
# #     plt.plot(est - exp_data[i,:])
# #     plt.show()
#
#
# # X_train, X_test, y_train, y_test = train_test_split(p1.T, exp_data.T, test_size=0.34, random_state= 10)
# # las = Lasso()
# # las.fit(X_train, y_test)
# # y_est = las.predict(X_test)
# #
# # print(las.coef_)

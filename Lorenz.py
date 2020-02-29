import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import csv
from sklearn.linear_model import lasso_path
from scipy import optimize
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

def lorenz(t, pop):
    alpha, beta, pho = 10, 8/3, 28
    x, y, z = pop
    return [alpha*(y-x),
            x*(pho-z) - y,
            x*y - beta*z]

tspan = np.linspace(0.001, 100, num=100000)
dt = 0.001
ini = [-8, 7, 27]
sol = integrate.solve_ivp(lorenz, [tspan[0], tspan[-1]], ini, method='RK45', t_eval=tspan)


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


theta = p1.T
exp_data = exp_data.T
coeff_ = np.dot(np.linalg.pinv(theta), exp_data)


def sparsifyDynamics(theta, exp_data, lambda1):
    for k in range(1000):
        small_idx = (abs(coeff_) < lambda1)
        big_idx = (abs(coeff_) >= lambda1)
        coeff_[small_idx] = 0
        for i in range(len(exp_data[0])):
            coeff_[big_idx[:, i], i] = np.dot(np.linalg.pinv(theta[:, big_idx[:, i]]), exp_data[:, i])
    return coeff_

coeff_ = sparsifyDynamics(theta, exp_data, 0.05)


print(coeff_)
est = np.dot(p1.T, coeff_)
#
plt.plot(est[:,0], '-*', lw=3, color='salmon', label ='est_dR')
plt.plot(exp_data[:,0], color='red', label ='true_dR')
plt.plot(est[:,1],'-*', lw=3, color='skyblue', label ='est_dS')
plt.plot(exp_data[:,1], color='blue', label = 'true_dS')
plt.plot(est[:,2], '-*',lw=3, color='grey', label = 'est_dM')
plt.plot(exp_data[:,2], color='black', label = 'true_dM')
plt.legend()
plt.show()
# print(est)

# csvfile = "./output2.csv"
# with open(csvfile, "w") as output2:
#     writer = csv.writer(output2, lineterminator='\n')
#     writer.writerows(coeff_)
#
#

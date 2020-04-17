import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from utils import *
import csv
from sklearn.linear_model import lasso_path
from scipy import optimize
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

def RSM(t, pop):
    a_r, a_s, a_m, K, e, f = 1.1, 2.0, 1.1, 10**4, 0.01, 0.01
    R, S, M = pop
    return [a_r*R*(1-(R+S+M)/K),
            a_s*S*(1-(R+S+M)/K) - e*S*R - f*S*M,
            a_m*M*(1-(R+S+M)/K) + e*S*R + f*S*M]

tspan = np.linspace(0, 20, num=2001)
dt = 0.01
ini = [10, 10000, 0]
sol = integrate.solve_ivp(RSM, [tspan[0], tspan[-1]], ini, method='RK45', t_eval=tspan)

sol_dx = np.array(RSM(sol.t, sol.y))
term_lib, term_des = lib_terms(sol.y, 4, "")
# for i in range(sol_dx.shape[0]):
#     term_lib = np.hstack((term_lib, term_lib * sol_dx[[i]].T))

num = 3
lambda_ = 1e-8
MaxIter = 10000
dic_lib = {}
dic_Xi = {}
dic_num = {}
dic_error = {}
dic_lambda = {}
i = 0

while True:
    coeff_ = sparsifyDynamics(term_lib, sol_dx.T, lambda_)
    dic_Xi[i] = coeff_
    temp_ = np.matmul(term_lib, dic_Xi[i])
    err_tmp = eucdist_2D(temp_, sol_dx.T)
    dic_error[i] = err_tmp
    dic_lambda[i] = lambda_
    lambda_ *= 2
    i += 1
    if lambda_ > 0.5:
        break



# plt.figure(figsize=(10,6))
# plt.subplot(1,2,1)
# plt.scatter(log_lambda_vec, terms_vec)
# plt.xlabel("Threshold (log_$\lambda$)")
# plt.ylabel("Number of terms")
# plt.subplot(1,2,2)
# plt.scatter(terms_vec, log_err_vec)
# plt.xlabel("Number of terms")
# plt.ylabel("Error (log)")
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import linalg
from utils import *

# system size
n = 1

# default value
Vmax, Km, k = 1.5, 0.3, 0.6
tspan = np.linspace(0.01, 4, num=400)
dt = 0.01
ini = [0.5]


sol = integrate.solve_ivp(fun=lambda t, x: MMKinetics(t, x, Vmax, Km, k),
                          t_span=[tspan[0], tspan[-1]], y0=ini, method='RK45', t_eval=tspan)


plt.plot(sol.y[0])
plt.show()

sol_dx = MMKinetics(sol.t, sol.y, Vmax, Km, k)
plt.show()
term_lib, term_des = lib_terms(sol.y, 6, "")
term_lib = np.hstack((term_lib, term_lib * sol_dx.T))


tol, pflag = 1e-5,1
dic_Xi, dic_lib, dic_lambda, dic_num, dic_error = ADMpareto(term_lib,tol, pflag)
lambda_vec = list(dic_lambda.values())
terms_vec = list(dic_num.values())
err_vec = list(dic_error.values())
log_err_vec = np.log(err_vec)
log_lambda_vec = np.log(lambda_vec)

# plot
plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.scatter(log_lambda_vec, terms_vec)
plt.xlabel("Threshold (log_$\lambda$)")
plt.ylabel("Number of terms")
plt.subplot(1,2,2)
plt.scatter(terms_vec, log_err_vec)
plt.xlabel("Number of terms")
plt.ylabel("Error (log)")
plt.show()

# def new_MMKinetics(term_lib, dic_Xi, n):
#     half_count = int(term_lib.shape[1] /2)
#     total_xi = len(dic_Xi)
#     #term_coff = dic_Xi[total_xi - n]
#     term_coff = [0.18, -0.9, 0, 0, 0.3, 1, 0, 0]
#     print(term_coff)
#     return -np.divide(np.matmul(term_lib[:,:half_count], term_coff[:half_count]),
#                       np.matmul(term_lib[:,half_count:], term_coff[half_count:]))
#
# res2 = new_MMKinetics(term_lib, dic_Xi, 4)
# plt.plot(sol_dx[0])
# plt.plot(res2)
# plt.show()
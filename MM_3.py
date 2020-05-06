import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import linalg
from utils import *

# system size
n = 1


def MMKinetics(t, x):
    Vmax, Km, k = 1.5, 0.3, 0.6
    return k - np.divide(Vmax*x, (Km+x))

tspan = np.linspace(0.01, 4, num=400)
dt = 0.01
ini = [0.5]
sol = integrate.solve_ivp(MMKinetics, [tspan[0], tspan[-1]], ini, method='RK45', t_eval=tspan)


plt.plot(sol.y[0])
plt.show()

sol_dx = MMKinetics(sol.t, sol.y)
plt.show()
term_lib, term_des = lib_terms(sol.y, 6, "")
term_lib = np.hstack((term_lib, term_lib * sol_dx.T))


tol, pflag = 1e-5,1
dic_Xi, dic_lib, dic_lambda, dic_num, dic_error = ADMpareto(term_lib,tol)
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

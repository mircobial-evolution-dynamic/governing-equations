import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import linalg
from sklearn.preprocessing import normalize
import csv
from sklearn.linear_model import lasso_path
# system size
n = 1

# parameters


def MMKinetics(t, x):
    Vmax, Km, k = 1.5, 0.3, 0.6
    return k - np.divide(Vmax*x, (Km+x))

def newmm(t,x):
    k1, k2, k3, k4 = 0.1295, -0.6474, 0.2158, 0.7194
    return np.divide((k1 + k2*x), (k3 + k4*x))

tspan = np.linspace(0.01, 4, num=400)
dt = 0.01
ini = [0.5]
sol = integrate.solve_ivp(MMKinetics, [tspan[0], tspan[-1]], ini, method='RK45', t_eval=tspan)
sol2 = integrate.solve_ivp(newmm, [tspan[0], tspan[-1]], ini, method='RK45', t_eval=tspan)
plt.plot(sol.y[0],linewidth=2, label = 'Ground Truth', color = 'orange')
plt.plot(sol2.y[0],linestyle='dashed', label = 'Identified Value', color = 'black')
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel('Time Steps', fontsize = 18)
plt.ylabel('Substrate Concentration', fontsize = 18)
plt.legend(fontsize = 18)
plt.show()

sol_dx = MMKinetics(sol.t, sol.y)
plt.show()
exp_data = sol_dx
def data_aug(sol_):
    n = len(sol_)
    m = sol_.size
    for i in range(n):
        sol_ = np.vstack((sol_, sol_[i]**2))
    for i in range(n):
        for j in range(i+1, n):
            sol_ = np.vstack((sol_, sol_[i]*sol_[j]))
    for i in range(n):
        for j in range(n, 2*n):
            sol_ = np.vstack((sol_, sol_[i]*sol_[j]))
    sol_ = np.vstack((np.ones((1, m)), sol_))
    return sol_


def data_derivative(sol_, d_sol_):
    n = len(sol_)
    for i in range(n):
        sol_ = np.vstack((sol_, np.multiply(sol_[i], d_sol_)))
    return sol_


term_lib = data_aug(sol.y)
term_lib = data_derivative(term_lib, sol_dx).T


def soft_thresholding(X, lambda_):
    temp_compare = X.T.copy()
    for i in range(len(X)):
        if abs(X[i]) - lambda_ < 0:
            temp_compare[:,[i]]= 0
        else:
            temp_compare[:, [i]] = abs(X[i]) - lambda_
    # tmp_compare = X[np.where(abs(X) - lambda_ > 0)]
    # tmp_compare = np.expand_dims(tmp_compare, axis =1)
    return np.multiply(np.sign(X), temp_compare.T)


def ADM(lib_null,q_init,lambda_,MaxIter, tol):
    q = q_init.copy()
    for i in range(MaxIter):
        q_old = q.copy()
        x = soft_thresholding(lib_null @ q_init, lambda_)
        temp_ = lib_null.T @ x
        q = temp_/ np.linalg.norm(temp_, 2)
        res_q = np.linalg.norm(q_old - q ,2)

        if res_q <= tol:
            return q


def ADMinitvary(lib_null, lambda_, MaxIter, tol, pflag):
    lib_null_norm = lib_null.copy()
    for i in range(len(lib_null[0])):
        lib_null_norm[:,i] = lib_null[:,i]/lib_null[:,i].mean()

    q_final = np.empty_like(lib_null.T)
    out = np.zeros((len(lib_null), len(lib_null)))
    nzeros = np.zeros((1, len(lib_null)))
    for i in range(len(lib_null_norm)):
        q_ini = lib_null_norm[[i],:].T
        temp_q = ADM(lib_null, q_ini, lambda_, MaxIter, tol)
        q_final[:,[i]] = temp_q
        temp_out = lib_null @ temp_q
        out[:,[i]] = temp_out
        nzeros_temp = sum(list((abs(temp_out) < lambda_)))
        nzeros[:,[i]] = float(nzeros_temp)
    idx_sparse = np.where(nzeros == max(np.squeeze(nzeros)))[0]
    ind_lib = np.where(abs(out[:, idx_sparse[0]]) >= lambda_)[0]
    Xi = out[:, idx_sparse[0]]
    small_idx = np.where(abs(out[:, idx_sparse[0]]) < lambda_)[0]
    Xi[small_idx] = 0
    numterms = len(ind_lib)
    return ind_lib, Xi, numterms

def ADMpareto(term_lib, tol, pflag):
    lib_null = linalg.null_space(term_lib)
    num = 1
    lambda_ = 1e-8
    MaxIter = 10000
    dic_lib = {}
    dic_Xi = {}
    dic_num = {}
    dic_error = {}
    dic_lambda = {}
    ii = 0
    while num > 0:
        temp_ind_lib, temp_Xi, temp_numterms = ADMinitvary(lib_null, lambda_, MaxIter, tol, pflag)
        dic_lib[ii] = temp_ind_lib
        dic_Xi[ii] = temp_Xi
        dic_num[ii] = temp_numterms

        error_temp = sum(np.matmul(term_lib, dic_Xi[ii]))
        dic_error[ii] = error_temp
        dic_lambda[ii] = lambda_
        lambda_ *= 2
        num = dic_num[ii]
        ii += 1
        if lambda_ > 0.5:
            break

    return dic_Xi, dic_lib, dic_lambda, dic_num, dic_error


tol, pflag = 1e-5,1
dic_Xi, dic_lib, dic_lambda, dic_num, dic_error = ADMpareto(term_lib,tol, pflag)
lambda_vec = list(dic_lambda.values())
terms_vec = list(dic_num.values())
err_vec = list(dic_error.values())
log_err_vec = np.log10(err_vec)
log_lambda_vec = np.log10(lambda_vec)

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.scatter(log_lambda_vec, terms_vec)
plt.xlabel("Threshold (log_$\lambda$)", fontsize = 18)
plt.ylabel("Number of terms", fontsize =18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.subplot(1,2,2)
plt.scatter(terms_vec, log_err_vec)
plt.xlabel("Number of terms", fontsize =18)
plt.ylabel("Error (log)", fontsize =18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.subplots_adjust(wspace=0.5)
plt.show()

def new_MMKinetics(term_lib, dic_Xi, n):
    half_count = int(term_lib.shape[1] /2)
    total_xi = len(dic_Xi)
    #term_coff = dic_Xi[total_xi - n]
    term_coff = [0.18, -0.9, 0, 0, 0.3, 1, 0, 0]
    print(term_coff)
    return -np.divide(np.matmul(term_lib[:,:half_count], term_coff[:half_count]),
                      np.matmul(term_lib[:,half_count:], term_coff[half_count:]))


res2 = new_MMKinetics(term_lib, dic_Xi, 4)
plt.plot(sol_dx[0])
plt.plot(res2)
plt.show()


###########MM noise and denoise ############
####add noise####
eps = 0.01  #float; magnitude of noise
sol_noise_y = np.ndarray(shape=(sol.y.shape))
for i in range(0,sol.y.shape[0]):
    SIZE = np.size(sol.y[i])
    sol_noise_y[i] = sol.y[i] + eps*np.random.normal(0,1,(SIZE))
    

#denoise
from scipy import signal
x0=sol_noise_y[0]
b,a = signal.butter(3,0.05)
zi = signal.lfilter_zi(b,a)
z,_=signal.lfilter(b,a,x0,zi=zi*x0[0])
z2,_=signal.lfilter(b,a,z,zi=zi*z[0])
y=signal.filtfilt(b,a,x0)

# plt.figure
# plt.plot(x0,'b')
# plt.plot(z,'r--',z2,'r',y,'k')
# plt.legend(('noisy signal', 'lfilter, once','lfilter, twice','filtfilt'),loc='best')
# plt.show


yy=y.reshape(1,400)

sol_dx = MMKinetics(sol.t, yy)
exp_data = sol_dx
term_lib = data_aug(yy)
term_lib = data_derivative(term_lib, sol_dx).T

tol, pflag = 1e-5,1
dic_Xi, dic_lib, dic_lambda, dic_num, dic_error = ADMpareto(term_lib,tol, pflag)
lambda_vec = list(dic_lambda.values())
terms_vec = list(dic_num.values())
err_vec = list(dic_error.values())
log_err_vec = np.log10(err_vec)
log_lambda_vec = np.log10(lambda_vec)

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.scatter(log_lambda_vec, terms_vec)
plt.xlabel("Threshold (log_$\lambda$)", fontsize = 18)
plt.ylabel("Number of terms", fontsize =18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.subplot(1,2,2)
plt.scatter(terms_vec, log_err_vec)
plt.xlabel("Number of terms", fontsize =18)
plt.ylabel("Error (log)", fontsize =18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.subplots_adjust(wspace=0.5)
plt.show()

print(terms_vec)
dic_Xi[23]
def newmm(t,x):
    k1, k2, k3, k4 = 0.1295, -0.6479, 0.2158, 0.7190
    return np.divide((k1 + k2*x), (k3 + k4*x))

sol2 = integrate.solve_ivp(newmm, [tspan[0], tspan[-1]], ini, method='RK45', t_eval=tspan)


plt.figure(figsize=(6,4))
plt.plot(x0,'y')
plt.plot(z,'r--',z2,'r',y,'b.',sol2.y[0],'k--')
plt.legend(('Noisy Truth', 'lfilter, once','lfilter, twice','Denoised Truth','Identified Value'),loc='best',fontsize=16)
plt.xlabel('Time Steps', fontsize = 18)
plt.ylabel('Substrate Concentration', fontsize = 18)
# plt.legend(fontsize = 18)
plt.show
# plt.savefig('MM_fig3.png')
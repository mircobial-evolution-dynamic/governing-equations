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


def data_derivative(sol_,d_sol_):
    n = len(sol_)
    for i in range(n):
        sol_ = np.vstack((sol_, np.multiply(sol_[i], d_sol_)))
    return sol_


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


#def optimal_SVHT_coef
def IfElse(Q,point,counterPoint):
    #if Q, POINT, COUNTERPOINT ARE arrays???; len(point) should equal or greater than len(Q)!
    y=point;
    P=np.zeros(len(Q))
    # P is 'not Q or ~Q'
    for q in range(len(Q)):
        if Q[q]==0:
            P[q]=1
    if len(counterPoint)==1:
        counterPoint = counterPoint*np.ones(len(Q))

    for p in range(len(P)):
            
        if P[p]==1:
            y[p]=counterPoint[p]
    
    return y

def Fun_MarPas(x,beta,gamma):
    topSpec = (1 + np.sqrt(beta))**2
    botSpec = (1 - np.sqrt(beta))**2
    Q=(topSpec-x)*(x-botSpec)
    for q in range(len(Q)):
        if Q[q]>0:
            Q[q]=1
        else:
            Q[q]=0
    point = np.sqrt((topSpec-x)*(x-botSpec))/(beta*x)/(2*np.pi)  
    InitialMar = IfElse(Q,point,np.array([0])) #'0' number used to replace others
    if np.all(gamma != 0):
        return np.power(list(x),list(gamma))*InitialMar
    else:
        return InitialMar
    
    
    
#### separate
def MarPas(x,beta):
    topSpec = (1 + np.sqrt(beta))**2
    botSpec = (1 - np.sqrt(beta))**2
    Q=(topSpec-x)*(x-botSpec)
    for q in range(len(Q)):
        if Q[q]>0:
            Q[q]=1
        else:
            Q[q]=0
    point = np.sqrt((topSpec-x)*(x-botSpec))/(beta*x)/(2*np.pi)  
    InitialMar = IfElse(Q,point,np.array([0])) #########9#####0
#     if np.all(gamma != 0):
#         return np.power(list(x),list(gamma))*InitialMar
#     else:
#         return InitialMar
    return InitialMar

def incMarPas(x0,beta,gamma):
    #beta is an array including one (****or more****not true) number; 0<beta<=1 
    #x0 is an array with multiple numbers!!!
#     for b in beta: 
#         if b > 1:
#             raise Exception('dimension error')
    topSpec = (1 + np.sqrt(beta))**2
#     botSpec = (1 - np.sqrt(beta))**2
    x2=lambda x: Fun_MarPas(x,beta,gamma)
    I1 = np.zeros([len(x0)])#I1 is integrate collection
    I2 = np.zeros([len(x0)])
    for i in range(len(x0)):
        X0=x0[i]
        RES = integrate.quad(x2,np.array([X0]), topSpec)
        I1[i] = RES[0]
        I2[i] = RES[1]
#     return integrate.quad(x2, x0, topSpec)  #the first part of return is integrate
    return I1

#if inputs are arrarys including multiple numbers?
def incMarPas2(x0,beta,gamma):
    #beta is an array including one or more numbers; 0<beta<=1
#     for b in beta: 
#         if b > 1:
#             raise Exception('dimension error')
    topSpec = (1 + np.sqrt(beta))**2   #is an array including multiple numbers if beta is an array including multiple numbers
        

    I1 = np.zeros([len(beta)])
    I2 = np.zeros([len(beta)])
    for i in range(len(beta)):
        BETA = beta[i]
        GAMMA = gamma[i]
        x2=lambda x: Fun_MarPas(x,np.array([BETA]),np.array([GAMMA]))
        X0=x0[i]
        TOPSPEC=topSpec[i]
        RES = integrate.quad(x2,np.array([X0]), np.array([TOPSPEC]))
        I1[i] = RES[0]
        I2[i] = RES[1]
    return I1, I2

def MedianMarcenkoPastur(beta):
    #beta should be one number array!!!here
#     MarPas = lambda x: 1-incMarPas(x,beta,np.array([0]))
    lobnd = (1-np.sqrt(beta))**2
    hibnd = (1+np.sqrt(beta))**2
    change = 1
    while change and (hibnd - lobnd > 0.001):
        change = 0;
        x = np.linspace(start = lobnd, stop = hibnd, num = 5)
        y = np.zeros(len(x))
        for i in range(len(x)):
            y[i] = 1-incMarPas(x[i],beta,0)
        if any(y<0.5):
            lobnd = max(x[y < 0.5])
            change = 1
        if any(y>0.5):
            hibnd = min(x[y > 0.5])
            change = 1
    return (hibnd+lobnd)/2

def optimal_SVHT_coef_sigma_known(beta):
    #lambda_star
    #omit ensuring beta
    w = (8*beta)/(beta+1+np.sqrt(beta**2+14*beta+1))
    lambda_star = np.sqrt(2*(beta+1)+w)
    return lambda_star

def optimal_SVHT_coef_sigma_unknown(beta):
    coef = optimal_SVHT_coef_sigma_known(beta)
    MPmedian = np.zeros(len(beta))
    for i in range(len(beta)):
        MPmedian[i] = MedianMarcenkoPastur(beta)
    return coef/np.sqrt(MPmedian)

def optimal_SVHT_coef(beta,noiselevel):
    if noiselevel == 0:
        return optimal_SVHT_coef_sigma_unknown(beta)
    else:
        return optimal_SVHT_coef_sigma_known(beta)
    
    
    
    
# #add noise reduction in ADMpareto; replace previous ADMpareto
# def ADMpareto(term_lib, tol, pflag):
#     #####
#     u, s, vh = np.linalg.svd(term_lib, full_matrices=False)
#     m,n = term_lib.shape  #m/n aspect ratio of matrix to be denoised?
#     ydi = np.diag(term_lib)
#     beta = np.array([n/m])
#     threshold = optimal_SVHT_coef(beta,0)[0]
#     ydi2 = np.copy(ydi)
#     ydi2[ydi2 < threshold*np.median(ydi2)] = 0
#     term_lib2 = np.linalg.multi_dot([u,np.diag(ydi2),vh.T])  #####no .T?
#     #####
#     lib_null = linalg.null_space(term_lib2) #2
#     num = 1
#     lambda_ = 1e-8
#     MaxIter = 10000

#     dic_lib = {}
#     dic_Xi = {}
#     dic_num = {}
#     dic_error = {}
#     dic_lambda = {}
#     ii = 0
#     while num > 0:
#         temp_ind_lib, temp_Xi, temp_numterms = ADMinitvary(lib_null, lambda_, MaxIter, tol, pflag)
#         dic_lib[ii] = temp_ind_lib
#         dic_Xi[ii] = temp_Xi
#         dic_num[ii] = temp_numterms

#         error_temp = sum(np.matmul(term_lib, dic_Xi[ii]))
#         dic_error[ii] = error_temp
#         dic_lambda[ii] = lambda_
#         lambda_ *= 2
#         num = dic_num[ii]
#         ii += 1
#         if lambda_ > 0.5:
#             break

#     return dic_Xi, dic_lib, dic_lambda, dic_num, dic_error



#add noise reduction in ADMpareto; replace previous ADMpareto
def ADMpareto(term_lib, tol, pflag):
    #####
    u, s, vh = np.linalg.svd(term_lib.T, full_matrices=False)
    m,n = term_lib.T.shape  #m/n aspect ratio of matrix to be denoised?
    ydi = np.diag(term_lib.T)
    beta = np.array([m/n])
    threshold = optimal_SVHT_coef(beta,0)[0]
    ydi2 = np.copy(ydi)
    ydi2[ydi2 < threshold*np.median(ydi2)] = 0
    term_lib2 = np.linalg.multi_dot([u,np.diag(ydi2),vh]).T  #####no .T?
    #####
    lib_null = linalg.null_space(term_lib2) #2
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



# system size
n = 1

# parameters

tspan = np.linspace(0.01, 4, num=400)
dt = 0.01
ini = [0.5]
sol = integrate.solve_ivp(MMKinetics, [tspan[0], tspan[-1]], ini, method='RK45', t_eval=tspan)#solver; np.size=400; (400,)
sol2 = integrate.solve_ivp(newmm, [tspan[0], tspan[-1]], ini, method='RK45', t_eval=tspan)


sol_dx = MMKinetics(sol.t, sol.y)
# plt.show()
exp_data = sol_dx  #shape:(1,400)


#add noise to 'x'
eps = 1e-4  #float; magnitude of noise
SIZE = np.size(sol.y[0])
sol_noise=sol.y+eps*np.random.normal(0,1,(SIZE))#np.size=400; shape=(1,400); type=numpy.ndarray
#calculate the corresponding derivative
sol_noise_dx = MMKinetics(sol.t, sol_noise)

# #test noise-added data
# sol.y = sol_noise
# sol_dx = sol_noise_dx
# # sol_dx = sol_dx_noise  # does not work
term_lib = data_aug(sol_noise)
term_lib = data_derivative(term_lib, sol_noise_dx).T#x,x^2,dx*x...; ## transpose ## 400*8

# term_lib = data_aug(sol.y)
# term_lib = data_derivative(term_lib, sol_dx).T#x,x^2,dx*x...; ## transpose ## 400*8

tol, pflag = 1e-5,1
dic_Xi, dic_lib, dic_lambda, dic_num, dic_error = ADMpareto(term_lib,tol, pflag)
lambda_vec = list(dic_lambda.values())
terms_vec = list(dic_num.values())
err_vec = list(dic_error.values())
log_err_vec = np.log10(err_vec)
log_lambda_vec = np.log10(lambda_vec)

print(dic_Xi)

plt.subplot(1,2,1)
plt.scatter(log_lambda_vec, terms_vec)
plt.xlabel("Threshold (log_$\lambda$)")
plt.ylabel("Number of terms")
plt.subplot(1,2,2)
plt.scatter(terms_vec, log_err_vec)
plt.xlabel("Number of terms")
plt.ylabel("Error (log)")
plt.show()
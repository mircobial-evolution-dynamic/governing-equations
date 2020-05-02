import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import csv
from scipy import optimize
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


from numpy import linalg as LA
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix
import itertools
from itertools import combinations
import operator
from mpl_toolkits import mplot3d

from utils import *



#####SEIR WITHOUT AIC#####
def SEIR(t, x):
    mu, beta, alpha, gamma, Ntot = 0, 0.3, 0.4, 0.04, 1e4
    S,E,I = x
    return np.array([mu-beta*S*I/Ntot-mu*S,
            beta*S*I/Ntot-(mu+alpha)*E,
            alpha*E - (mu+gamma)*I])

Ntot = 1e4
tspan = np.linspace(0, 250, num=2001)
dt = 0.01
ini = [0.99*Ntot, 0.01*Ntot, 0]
sol = integrate.solve_ivp(SEIR, [tspan[0], tspan[-1]], ini, method='RK45', t_eval=tspan)

plt.figure(figsize=(6,4))
plt.plot(sol.t, sol.y[0].T, linewidth=3, label='S')
plt.plot(sol.t, sol.y[1].T, linewidth=3,label='E')
plt.plot(sol.t, sol.y[2].T, linewidth=3, label='I')
plt.xlabel("Time Steps",fontsize=18)
plt.ylabel("State Variables",fontsize=18)
plt.legend(fontsize=18)
plt.show()


dx = SEIR(sol.t, sol.y)
theta, descr = lib_terms(sol.y,3,'SEI')
## theta and descr


#Euclidean distance
L,E,N = evaluate(theta,-10,1,30,dx.T)
print(L)
print(E)
print(N)

plt.subplot(1,2,1)
# plt.scatter(lambda_vec, num_terms)
plt.scatter(L, N)
plt.xlabel("Threshold ($\lambda$)",fontsize=18)
plt.ylabel("Number of terms",fontsize=18)
plt.subplot(1,2,2)
# plt.scatter(num_terms, eudist_vec)
plt.scatter(N, np.log(E))
plt.xlabel("Number of terms",fontsize=18)
plt.ylabel("log Euclidean distance",fontsize=18)
plt.subplots_adjust(wspace=0.5)
# plt.title('SEIR with Euclidean Distance')
plt.show()


###AIC etc###
def error(matrix1, matrix2):
    #return sum (abs(matrix1_element-matrix2_element)^2)/N 
    m,n = matrix1.shape
    m1,n1 = matrix2.shape
    if m != m1 or n != n1:
        import os, warnings
        warnings.warn('dimession conflict')
    eucdist = 0
    for i in range(n):
        eucdist = eucdist + np.linalg.norm(matrix1[:,i]-matrix2[:,i])
        
    return eucdist/(m*n)



def evaluate_AIC(theta, lambdastart, lambdaend, numlambda, dx):
    # return lambda vector, AIC vector (as the evaluation index), and num of terms
    import math
    from numpy import log as ln
    m,n = theta.shape  #m is time steps
    lambda_vec = np.logspace(lambdastart, lambdaend, numlambda)
    AIC_vec = []
    AICc_vec = []
    BIC_vec = []
    num_terms = []
    for i in lambda_vec:
        Xi = sparsifyDynamics(theta, dx, i)
        b = np.nonzero(Xi)
        p = len(np.unique(b[0]))
        sim_dx = np.matmul(theta,Xi)
#         aic = -math.log10(error(dx,sim_dx))+2*p
        aic = n*ln(error(dx,sim_dx)/n)+2*p
        aic_c = aic+2*p*(p+1)/(m-p-1)
        bic = aic - 2*p + 2*p*math.log(m)
        AIC_vec.append(aic)
        AICc_vec.append(aic_c)
        BIC_vec.append(bic)
        num_terms.append(np.count_nonzero(Xi))
    AIC_vec = np.array(AIC_vec)
    num_terms = np.array(num_terms)
    return lambda_vec, AIC_vec, AICc_vec, BIC_vec, num_terms


L,AIC,AICc, BIC, N = evaluate_AIC(theta,-10,1,20,dx.T)
print(L)
print(AIC)
print(N)

plt.figure(figsize=(8,6))
plt.subplot(2,2,1)
# plt.scatter(lambda_vec, num_terms)
plt.scatter(L, N)
plt.xlabel("Threshold ($\lambda$)",fontsize=18)
plt.ylabel("# of terms",fontsize=18)
plt.subplot(2,2,2)
# plt.scatter(num_terms, eudist_vec)
plt.scatter(N, AIC)
plt.xlabel("Number of terms",fontsize=18)
plt.ylabel("AIC",fontsize=18)
plt.subplot(2,2,3)
# plt.scatter(num_terms, eudist_vec)
plt.scatter(N, AICc)
plt.xlabel("Number of terms",fontsize=18)
plt.ylabel("AICc",fontsize=18)
plt.subplot(2,2,4)
# plt.scatter(num_terms, eudist_vec)
plt.scatter(N, BIC)
plt.xlabel("Number of terms",fontsize=18)
plt.ylabel("BIC",fontsize=18)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
# plt.title('SEIR with Euclidean Distance')
plt.show()



def evaluate_NEW(theta, lambdastart, lambdaend, numlambda, dx):
    # return lambda vector, NEW vector (as the evaluation index), and num of terms
    import math
    m,n = theta.shape  #m is time steps
    lambda_vec = np.logspace(lambdastart, lambdaend, numlambda)
    new_vec = []
    num_terms = []
    for i in lambda_vec:
        Xi = sparsifyDynamics(theta, dx, i)
        b = np.nonzero(Xi)
        p = len(np.unique(b[0]))
        sim_dx = np.matmul(theta,Xi)
#         aic = -math.log10(error(dx,sim_dx))+2*p
        new_value = math.log(eucdist_2D(dx,sim_dx))+p/2
        new_vec.append(new_value)
        num_terms.append(np.count_nonzero(Xi))
    new_vec = np.array(new_vec)
    num_terms = np.array(num_terms)
    return lambda_vec, new_vec, num_terms


L, NEW_VEC, N = evaluate_NEW(theta,-10,1,20,dx.T)
# print(L)
# print(NEW_VEC)
# print(N)
## plot new vector of SEIR
plt.scatter(N, NEW_VEC)
plt.xlabel("Number of terms",fontsize=18)
plt.ylabel("New Index",fontsize=18)
plt.subplots_adjust(wspace=0.5)
# plt.title('SEIR with Euclidean Distance')
plt.show()

#########################################################################################
##RSM Model##
def RSM(t, pop):
#     a_r, a_s, a_m, K, e, f = 1.1, 2.0, 1.1, 10**3, 0.001, 0.001
    a_r, a_s, a_m, K, e, f = 2.5, 2.0, 2.5, 10**3, 0.001, 0.001
    # a_i is growth rate
    R, S, M = pop
    return np.array([a_r*R*(1-(R+S+M)/K),
            a_s*S*(1-(R+S+M)/K) - e*S*R - f*S*M,
            a_m*M*(1-(R+S+M)/K) + e*S*R + f*S*M])
            
tspan = np.linspace(0, 20, num=2001)
dt = 0.01
ini = [10, 100, 0]
sol = integrate.solve_ivp(RSM, [tspan[0], tspan[-1]], ini, method='RK45', t_eval=tspan)


plt.figure(figsize=(6,4))
plt.plot(sol.t, sol.y[0].T,linewidth=3, label='R')
plt.plot(sol.t, sol.y[1].T, linewidth=3,label='S')
plt.plot(sol.t, sol.y[2].T,linewidth=3, label='M')
plt.xlabel("Time Steps",fontsize=18)
plt.ylabel("State Variables",fontsize=18)
plt.legend()
plt.legend(fontsize=18)
plt.show()


dx = RSM(sol.t, sol.y)
theta, descr = lib_terms(sol.y,3,'RSM')
## theta and descr

L,E,N = evaluate(theta,-10,1,30,dx.T)
print(L)
print(E)
print(N)


plt.subplot(1,2,1)
# plt.scatter(lambda_vec, num_terms)
plt.scatter(L, N)
plt.xlabel("Threshold ($\lambda$)",fontsize=18)
plt.ylabel("Number of terms",fontsize=18)
plt.subplot(1,2,2)
# plt.scatter(num_terms, eudist_vec)
plt.scatter(N, np.log(E))
plt.xlabel("Number of terms",fontsize=18)
plt.ylabel("log Euclidean distance",fontsize=18)
plt.subplots_adjust(wspace=0.5)
# plt.title('SEIR with Euclidean Distance')
plt.show()

#AIC for RSM

L,AIC,AICc, BIC, N = evaluate_AIC(theta,-10,1,20,dx.T)
print(L)
print(AIC)
print(N)

plt.figure(figsize=(8,6))
plt.subplot(2,2,1)
# plt.scatter(lambda_vec, num_terms)
plt.scatter(L, N)
plt.xlabel("Threshold ($\lambda$)",fontsize=18)
plt.ylabel("# of terms",fontsize=18)
plt.subplot(2,2,2)
# plt.scatter(num_terms, eudist_vec)
plt.scatter(N, AIC)
plt.xlabel("Number of terms",fontsize=18)
plt.ylabel("AIC",fontsize=18)
plt.subplot(2,2,3)
# plt.scatter(num_terms, eudist_vec)
plt.scatter(N, AICc)
plt.xlabel("Number of terms",fontsize=18)
plt.ylabel("AICc",fontsize=18)
plt.subplot(2,2,4)
# plt.scatter(num_terms, eudist_vec)
plt.scatter(N, BIC)
plt.xlabel("Number of terms",fontsize=18)
plt.ylabel("BIC",fontsize=18)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
# plt.title('SEIR with Euclidean Distance')
plt.show()

L, NEW_VEC, N = evaluate_NEW(theta,-10,1,20,dx.T)
print(L)
print(NEW_VEC)
print(N)


plt.scatter(N, NEW_VEC)
plt.xlabel("Number of terms",fontsize=18)
plt.ylabel("New Index",fontsize=18)
plt.subplots_adjust(wspace=0.5)
# plt.title('SEIR with Euclidean Distance')
plt.show()



# ######################################################################################
# ##### Lorenz System with changed params ######
# def lorenz(t, pop):
#     # alpha, beta, pho = 10, 8/3, 28
#     alpha, beta, pho = 10, 2, 25
#     x, y, z = pop
#     return np.array([alpha*(y-x),
#             x*(pho-z) - y,
#             x*y - beta*z])

# tspan = np.linspace(0.001, 100, num=100000)
# dt = 0.001
# ini = [-8, 7, 27]
# sol = integrate.solve_ivp(lorenz, [tspan[0], tspan[-1]], ini, method='RK45', t_eval=tspan)


# # #fig1
# # plt.plot(sol.t, sol.y[0].T, 'k',label='x')
# # plt.plot(sol.t, sol.y[1].T, 'g',label='y')
# # plt.plot(sol.t, sol.y[2].T, 'y',label='z')
# # plt.legend()
# # plt.show()


# #fig2
# y1 = sol.y[0].T
# y2 = sol.y[1].T
# y3 = sol.y[2].T

# x = sol.t

# fig = plt.figure()
# plt.subplot(3,1,1)
# plt.plot(x,y1,'k')
# plt.title('Time Course Data',fontsize=20)
# plt.ylabel('x',fontsize=18)
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off

# plt.subplot(3,1,2)
# plt.plot(x,y2,'g')
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
# plt.ylabel('y',fontsize=18)


# plt.subplot(3,1,3)
# plt.plot(x,y3,'b')
# plt.ylabel('z',fontsize=18)
# plt.xlabel('time step',fontsize=18)
# plt.show()

# # plt.savefig('lorenz_timecourse.png')


# #fig3
# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# xdata = sol.y[0]
# ydata = sol.y[1]
# zdata = sol.y[2]
# # ax.scatter3D(xdata,ydata,zdata, c=zdata)
# ax.plot3D(xdata,ydata,zdata)
# ax.set_xlabel('x',fontsize=18)
# ax.set_ylabel('y',fontsize=18)
# ax.set_zlabel('z',fontsize=18)
# ax.set_title('Trajectory',fontsize=20)
# plt.show()


# # #### results ######
# # dx = lorenz(sol.t, sol.y)
# # theta, descr = lib_terms(sol.y,3,'xyz')
# # ## theta and descr
# # Xi = sparsifyDynamics(theta, dx.T, 0.05)
# # print(Xi)
# # # obtain Xi here by changing lambda
# # L,E,N = evaluate(theta,-5,1,10,dx.T)
# # print(L)
# # print(E)
# # print(N)


# # plt.subplot(1,2,1)
# # # plt.scatter(lambda_vec, num_terms)
# # plt.scatter(L, N)
# # plt.xlabel("Threshold ($\lambda$)")
# # plt.ylabel("Number of terms")
# # plt.subplot(1,2,2)
# # # plt.scatter(num_terms, eudist_vec)
# # plt.scatter(N, np.log(E))
# # plt.xlabel("Number of terms")
# # plt.ylabel("log Euclidean distance")
# # plt.title('SEIR with Euclidean Distance')
# # plt.show()




# ##### add noise and denoise for Lorenz
# eps = 1  #float; magnitude of noise
# sol_noise_y = np.ndarray(shape=(sol.y.shape))
# for i in range(0,sol.y.shape[0]):
#     SIZE = np.size(sol.y[i])
#     sol_noise_y[i] = sol.y[i] + eps*np.random.normal(0,1,(SIZE))
    
    
# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# xdata = sol_noise_y[0]
# ydata = sol_noise_y[1]
# zdata = sol_noise_y[2]
# # ax.scatter3D(xdata,ydata,zdata, c=zdata)
# ax.plot3D(xdata,ydata,zdata)
# ax.set_xlabel('x',fontsize=18)
# ax.set_ylabel('y',fontsize=18)
# ax.set_zlabel('z',fontsize=18)
# # ax.set_title('trajectory',fontsize=20)
# ax.set_title('noisy trajectory',fontsize=20)
# plt.show()

# ##noise reduction
# from scipy.signal import lfilter
# n=20
# b=[1.0/n]*n
# a=1
# sol_denoise_y = np.ndarray(shape=(sol.y.shape))
# for i in range(0,sol_noise_y.shape[0]):
#     sol_denoise_y[i]=lfilter(b,a,sol_noise_y[i])

# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# # xdata = sol_noise_y[0]
# # ydata = sol_noise_y[1]
# # zdata = sol_noise_y[2]
# xdata = sol_denoise_y[0]
# ydata = sol_denoise_y[1]
# zdata = sol_denoise_y[2]
# # ax.scatter3D(xdata,ydata,zdata, c=zdata)
# ax.plot3D(xdata,ydata,zdata)
# ax.set_xlabel('x',fontsize=18)
# ax.set_ylabel('y',fontsize=18)
# ax.set_zlabel('z',fontsize=18)
# ax.set_title('denoised trajectory',fontsize=20)
# plt.show()


# #results
# dx = lorenz(sol.t, sol_denoise_y)
# theta, descr = lib_terms(sol_denoise_y,3,'xyz')
# ## theta and descr
# Xi = sparsifyDynamics(theta, dx.T, 0.05)
# print(Xi)
# # obtain Xi here by changing lambda
# L,E,N = evaluate(theta,-5,1,10,dx.T)
# print(L)
# print(E)
# print(N)

# Xi = sparsifyDynamics(theta, dx.T, 1)
# print(Xi)

# # plt.title('SEIR with Euclidean Distance')
# plt.subplot(1,2,1)
# # plt.scatter(lambda_vec, num_terms)
# plt.scatter(L, N)
# plt.xlabel("Threshold ($\lambda$)",fontsize=18)
# plt.ylabel("Number of terms",fontsize=18)
# plt.subplot(1,2,2)
# # plt.scatter(num_terms, eudist_vec)
# plt.scatter(N, np.log(E))
# plt.xlabel("Number of terms",fontsize=18)
# plt.ylabel("log Euclidean distance",fontsize=18)
# plt.subplots_adjust(wspace=0.5)

# plt.show()


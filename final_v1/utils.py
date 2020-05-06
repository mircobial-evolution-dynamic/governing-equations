import numpy as np
import itertools
import operator

from scipy import linalg


# build library
def power_(d, order):
    # d is the number of variables; order of polynomials
    powers = []
    for p in range(1, order + 1):
        size = d + p - 1
        for indices in itertools.combinations(range(size), d - 1):  ##combinations
            starts = [0] + [index + 1 for index in indices]
            stops = indices + (size,)
            powers.append(tuple(map(operator.sub, stops, starts)))
    return powers


def lib_terms(data, order, description):
    # description is a list of name of variables, like [R, M, S]
    # description of lib
    descr = []
    # data is the input data, like R,M,S; order is the total order of polynomials

    d, t = data.shape  # d is the number of variables; t is the number of time points
    theta = np.ones((t, 1), dtype=np.float64)  # the first column of lib is '1'
    P = power_(d, order)
    for i in range(len(P)):
        new_col = np.zeros((t, 1), dtype=np.float64)
        for j in range(t):
            new_col[j] = np.prod(np.power(list(data[:, j]), list(P[i])))
        theta = np.hstack([theta, new_col.reshape(t, 1)])
        descr.append("{0} {1}".format(str(P[i]), str(description)))
    descr = ['1'] + descr

    return theta, descr



#ADM algrithm
def soft_thresholding(X, lambda_):
    temp_compare = X.T.copy()
    for i in range(len(X)):
        if abs(X[i]) - lambda_ < 0:
            temp_compare[:, [i]] = 0
        else:
            temp_compare[:, [i]] = abs(X[i]) - lambda_
    # tmp_compare = X[np.where(abs(X) - lambda_ > 0)]
    # tmp_compare = np.expand_dims(tmp_compare, axis =1)
    print(lambda_)
    return np.multiply(np.sign(X), temp_compare.T)


def ADM(lib_null, q_init, lambda_, MaxIter, tol):
    q = q_init.copy()
    for i in range(MaxIter):
        q_old = q.copy()
        x = soft_thresholding(lib_null @ q_init, lambda_)
        temp_ = lib_null.T @ x
        q = temp_ / np.linalg.norm(temp_, 2)
        res_q = np.linalg.norm(q_old - q, 2)

        if res_q <= tol:
            return q


def ADMinitvary(lib_null, lambda_, MaxIter, tol, pflag):
    lib_null_norm = lib_null.copy()
    for i in range(len(lib_null[0])):
        lib_null_norm[:, i] = lib_null[:, i] / lib_null[:, i].mean()

    q_final = np.empty_like(lib_null.T)
    out = np.zeros((len(lib_null), len(lib_null)))
    nzeros = np.zeros((1, len(lib_null)))
    for i in range(len(lib_null_norm)):
        q_ini = lib_null_norm[[i], :].T
        temp_q = ADM(lib_null, q_ini, lambda_, MaxIter, tol)
        q_final[:, [i]] = temp_q
        temp_out = lib_null @ temp_q
        out[:, [i]] = temp_out
        nzeros_temp = sum(list((abs(temp_out) < lambda_)))
        nzeros[:, [i]] = float(nzeros_temp)

    idx_sparse = np.where(nzeros == max(np.squeeze(nzeros)))[0]
    ind_lib = np.where(abs(out[:, idx_sparse[0]]) >= lambda_)[0]
    Xi = out[:, idx_sparse[0]]
    small_idx = np.where(abs(out[:, idx_sparse[0]]) < lambda_)[0]
    Xi[small_idx] = 0
    numterms = len(ind_lib)
    return ind_lib, Xi, numterms


def sparsifyDynamics(Theta, dx, Lambda):
    # theta.shape = 248*10 (time points*functions); dx.shape = 248*3 (time points*variables)
    # need to ensure size or dimenssions !!!
    #     dx = dx.T
    m, n = dx.shape  # (248*3)
    Xi = np.dot(np.linalg.pinv(Theta), dx)  # Xi.shape = 10*3
    # lambda is sparasification knob
    for k in range(10):  ###??
        small_idx = (abs(Xi) < Lambda)
        big_idx = (abs(Xi) >= Lambda)
        Xi[small_idx] = 0
    #     for i in range(n):
    #         #big_idx = np.bitwise_not([small_idx[:,i]]).T
    #         if dx == 0:
    #             Xi[big_idx[:,i],i] = linalg.null_space(Theta[:,big_idx[:,i]])
    #         else:
    #             Xi[big_idx[:,i],i] = np.dot(np.linalg.pinv(Theta[:,big_idx[:,i]]), dx[:, i])

    return Xi  # num of functions*num of variables



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


def data_derivative(sol_, d_sol_):
    n = len(sol_)
    for i in range(n):
        sol_ = np.vstack((sol_, np.multiply(sol_[i], d_sol_)))
    return sol_


def eucdist_2D(matrix1, matrix2):
    # return sum of euclidean distance of two matrices
    m, n = matrix1.shape
    m1, n1 = matrix2.shape
    if m != m1 or n != n1:
        import os, warnings
        warnings.warn('dimession conflict')
    eucdist = 0
    for i in range(n):
        eucdist = eucdist + np.linalg.norm(matrix1[:, i] - matrix2[:, i])

    return eucdist


def evaluate(theta, lambdastart, lambdaend, numlambda, dx):
    # return lambda vector, euclidean vector (as the evaluation index), and num of terms
    lambda_vec = np.logspace(lambdastart, lambdaend, numlambda)
    eucdist_vec = []
    num_terms = []
    for i in lambda_vec:
        Xi = sparsifyDynamics(theta, dx, i)
        sim_dx = np.matmul(theta, Xi)
        eucdist_vec.append(eucdist_2D(dx, sim_dx))
        num_terms.append(np.count_nonzero(Xi))
    eucdist_vec = np.array(eucdist_vec)
    num_terms = np.array(num_terms)
    return lambda_vec, eucdist_vec, num_terms


def MMKinetics(t, x, Vmax, Km, k):
    return k - np.divide(Vmax * x, (Km + x))

    ###

def xy_func(x,y):
    return x+y

import numpy as np
import itertools
import operator

from scipy import linalg


def power_(d, order):
    """
    helper function for lib_terms
    :param d: number of variables, dtype = int
    :param order: order of polynomials, dtype = int
    :return: tuple of powers
    """
    powers = []
    for p in range(1, order + 1):
        size = d + p - 1
        for indices in itertools.combinations(range(size), d - 1):  #combinations
            starts = [0] + [index + 1 for index in indices]
            stops = indices + (size,)
            powers.append(tuple(map(operator.sub, stops, starts)))
    return powers


def lib_terms(data, order, description):
    """
    generate the library for system identification
    :param data: input dynamic values over time, data.shape = (the number of variable, length of time ), dtype = array
    :param order: order of polynomial, dtype = int
    :param description: the variable , dtype = str
    :return theta: library candidates, data.shape =(length of time, the number of candidate), dtype = array
    :return descr: library description
    """
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
    """
    helper function for ADM
    :param X: simulated data = null space of library * coeff_
    :param lambda_: threshold in lasso regression
    :return: simulated data in where the small trivial term has been removed
    """
    temp_compare = X.T.copy()
    for i in range(len(X)):
        if abs(X[i]) - lambda_ < 0:
            temp_compare[:, [i]] = 0
        else:
            temp_compare[:, [i]] = abs(X[i]) - lambda_
    print(lambda_)
    return np.multiply(np.sign(X), temp_compare.T)


def ADM(lib_null, q_init, lambda_, MaxIter, tol):
    """
    helper function in the ADMinitvary
    :param lib_null: null space of library
    :param q_init: initial guess of normalized null space
    :param lambda_: threshold in lasso regression
    :param MaxIter: maximum iterative time
    :param tol: tolerance
    :return: normalized null space
    """
    q = q_init.copy()
    for i in range(MaxIter):
        q_old = q.copy()
        x = soft_thresholding(lib_null @ q_init, lambda_)
        temp_ = lib_null.T @ x
        q = temp_ / np.linalg.norm(temp_, 2)
        res_q = np.linalg.norm(q_old - q, 2)

        if res_q <= tol:
            return q


def ADMinitvary(lib_null, lambda_, MaxIter, tol):
    """
    helper function for ADMpareto
    :param lib_null: null space of library
    :param lambda_: threshold in lasso regression
    :param MaxIter: maximum iterative time
    :param tol: tolerance
    :return ind_lib: index of library that is the identified term
    :return Xi: identified term
    :return numterms: number of identified term
    """
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

def ADMpareto(term_lib, tol):
    """
    system identification - iterative method to find the best fit
    :param term_lib: library
    :param tol: tolerance
    :return: dictionaries - dic_Xi: calculated coefficient, dic_lib: selected library candidate
                            dic_lambda: corresponding lambda value, dic_error: corresponding error value
    """
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
        temp_ind_lib, temp_Xi, temp_numterms = ADMinitvary(lib_null, lambda_, MaxIter, tol)
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


def sparsifyDynamics(Theta, dx, Lambda):
    """
    sparse identification
    :param Theta: library
    :param dx: experimental data or simulated data which is as the standard to evaluate the identified system
    :param Lambda: the threshold in lasso regression
    :return: identified system
    """
    Xi = np.dot(np.linalg.pinv(Theta), dx)
    for k in range(10):
        small_idx = (abs(Xi) < Lambda)
        Xi[small_idx] = 0

    return Xi



def eucdist_2D(matrix1, matrix2):
    """
    helper function for eucdist_evaluate
    calculate the euclidean distance between the true system and identified system
    :param matrix1: true system
    :param matrix2: identified system
    :return: euclidean distance
    """
    m, n = matrix1.shape
    m1, n1 = matrix2.shape
    if m != m1 or n != n1:
        import os, warnings
        warnings.warn('dimensions conflict')
    eucdist = 0
    for i in range(n):
        eucdist = eucdist + np.linalg.norm(matrix1[:, i] - matrix2[:, i])

    return eucdist


def eucdist_evaluate(theta, lambdastart, lambdaend, numlambda, dx):
    """
    evaluate the identified system based on calculated euclidean distance
    :param theta: library
    :param lambdastart: smallest lambda value
    :param lambdaend: largest lambda value
    :param numlambda: number of lambda value
    :param dx: experimental data or simulated data from true system
    :return: lambda_vec: different lambda value, eucdist_vec: corresponding euclidean distance,
            num_terms: the number of identified terms
    """

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

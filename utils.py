class utils:
    ##build library
    def power(d,order):
    # d is the number of variables; order of polynomials
        powers = []
        for p in range(1,order+1):
            size = d + p - 1
            for indices in itertools.combinations(range(size), d-1):   ##combinations
                starts = [0] + [index+1 for index in indices]
                stops = indices + (size,)
                powers.append(tuple(map(operator.sub, stops, starts)))
        return powers

    def lib(data,order):
        #data is the input data - sol.y, like R,M,S; order is the total order of polynomials
        d,t = data.shape # d is the number of variables; t is the number of time points
        Theta = np.ones((t,1), dtype=np.complex64) # the first column of lib is '1'
        P = power(d,order)
        for i in range(len(P)):
            new_col = np.zeros((t,1),dtype=np.complex64)
            for j in range(t):
                new_col[j] = np.prod(np.power(list(data[:,j]),list(P[i])))
            Theta = np.hstack([Theta, new_col.reshape(t,1)])
        return Theta

    def lib(data,order,description):
        #description is a list of name of variables, like [R, M, S]
        #description of lib
        descr = []
        #data is the input data, like R,M,S; order is the total order of polynomials
    
        d,t = data.shape # d is the number of variables; t is the number of time points
        Theta = np.ones((t,1), dtype=np.complex64) # the first column of lib is '1'
        P = power(d,order)
        for i in range(len(P)):
            new_col = np.zeros((t,1),dtype=np.complex64)
            for j in range(t):
                new_col[j] = np.prod(np.power(list(data[:,j]),list(P[i])))
            Theta = np.hstack([Theta, new_col.reshape(t,1)])
            descr.append("{0} {1}".format(str(P[i]), str(description)))
        descr = ['1']+descr
        
        return Theta,descr
    ##

    ##ADM algrithm
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
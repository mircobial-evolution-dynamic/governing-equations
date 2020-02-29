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

    ##
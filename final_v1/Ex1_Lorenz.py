import utils
import numpy as np
from scipy import integrate

def lorenz(t, pop):
    alpha, beta, pho = 10, 8/3, 28
    x, y, z = pop
    return [alpha*(y-x),
            x*(pho-z) - y,
            x*y - beta*z]

tspan = np.linspace(0.001, 100, num=100000)
dt = 0.001
ini = [-8, 7, 27]
sol = integrate.solve_ivp(lorenz, [tspan[0], tspan[-1]], ini, method='RK45', t_eval=tspan)


theta, descr = utils.lib_terms(sol.y, 3, 'xyz')

dx = np.array(lorenz(sol.t, sol.y))

coeff_ = utils.sparsifyDynamics(theta, dx.T, 0.05)
print(coeff_)





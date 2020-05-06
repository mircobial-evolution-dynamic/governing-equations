import utils
import numpy as np
from scipy import integrate

# define lorenz system
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

theta, descr = utils.lib_terms(sol.y.T, 5, ['x', 'y', 'z'])

exp_data = np.diff(sol.y, axis=1) /dt
exp_data = exp_data.T

coeff_ = utils.sparsifyDynamics(theta, exp_data, 0.05)
print(coeff_)





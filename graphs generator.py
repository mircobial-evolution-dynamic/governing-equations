import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import csv
from sklearn.linear_model import lasso_path
from scipy import optimize
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d


###########
#lorenz system#
###########
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

#fig 1-3 time course data
plt.plot(sol.t, sol.y[0].T, label='x')
# plt.plot(sol.t, sol.y[1].T, label='y')
# plt.plot(sol.t, sol.y[2].T, label='z')
plt.legend()
plt.show()


plt.plot(sol.t, sol.y[1].T, label='y')
# plt.plot(sol.t, sol.y[2].T, label='z')
plt.legend()
plt.show()


plt.plot(sol.t, sol.y[2].T, label='z')
plt.legend()
plt.show()

#figure 4 trajectory plot
fig = plt.figure()
ax = plt.axes(projection = '3d')
xdata = sol.y[0]
ydata = sol.y[1]
zdata = sol.y[2]
# ax.scatter3D(xdata,ydata,zdata, c=zdata)
ax.plot3D(xdata,ydata,zdata)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('trajectory')
plt.show()


#figure5 trajectory with noise level 1e-4/1e-2/1
eps = 1  #float; magnitude of noise
sol_noise_y = np.ndarray(shape=(sol.y.shape))
for i in range(0,sol.y.shape[0]):
    SIZE = np.size(sol.y[i])
    sol_noise_y[i] = sol.y[i] + eps*np.random.normal(0,1,(SIZE))

fig = plt.figure()
ax = plt.axes(projection = '3d')
xdata = sol_noise_y[0]
ydata = sol_noise_y[1]
zdata = sol_noise_y[2]
# ax.scatter3D(xdata,ydata,zdata, c=zdata)
ax.plot3D(xdata,ydata,zdata)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('trajectory')
plt.show()


##figure denoise
from scipy.signal import lfilter
n=20
b=[1.0/n]*n
a=1
sol_denoise_y = np.ndarray(shape=(sol.y.shape))
for i in range(0,sol_noise_y.shape[0]):
    sol_denoise_y[i]=lfilter(b,a,sol_noise_y[i])

fig = plt.figure()
ax = plt.axes(projection = '3d')
xdata = sol_denoise_y[0]
ydata = sol_denoise_y[1]
zdata = sol_denoise_y[2]
# ax.scatter3D(xdata,ydata,zdata, c=zdata)
ax.plot3D(xdata,ydata,zdata)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('trajectory')
plt.show()
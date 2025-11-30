import numpy as np
import matplotlib.pyplot as plt

from rk4_modified import rk4_modified
import random
from matplotlib.gridspec import GridSpec

# ------
# Lorenz 
# ------

a = random.uniform(-1, 1)
b = random.uniform(-1, 1)
c = random.uniform(-1, 1)

y1_0 = np.array([a, b, c]); print(y1_0)     # random initial value 1
y2_0 = np.array([0.001, 0.001, 0.001])         # initial value 2
t0 = 0.0    # initial time
T = 100      # final time
dt = 0.01   # step size

beta = 8/3; sigma = 10; rho = 20
mu = 10**-5

lorenz = lambda t , x : np.array([
    sigma*(x[1] - x[0]),
    x[0]*(rho - x[2]) - x[1],
    x[0]*x[1] - beta*x[2]
])

lorenz_modifiedx = lambda t , x : np.array([
    sigma*(x[1] - x[0]) + mu*(lorenz(t, x)[0]-x[0]),
    x[0]*(rho - x[2]) - x[1],
    x[0]*x[1] - beta*x[2]
])

lorenz_modifiedy = lambda t , x : np.array([
    sigma*(x[1] - x[0]) ,
    x[0]*(rho - x[2]) - x[1] + mu*(lorenz(t, x)[1]-x[1]),
    x[0]*x[1] - beta*x[2]
])

lorenz_modifiedz = lambda t , x : np.array([
    sigma*(x[1] - x[0]) ,
    x[0]*(rho - x[2]) - x[1],
    x[0]*x[1] - beta*x[2] + mu*(lorenz(t, x)[2]-x[2]) 
])

X1 , X2, errors, t = rk4_modified(lorenz, lorenz_modifiedx, y1_0, y2_0, t0, T, dt)
print(errors[-1])

fig = plt.figure(figsize=(5,7)) # 1x2
gs = GridSpec(2, 1, figure=fig) # actual layout

# first row = 3d plots
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.plot(X1[0,:], X1[1,:], X1[2,:])
ax1.plot(X2[0,:], X2[1,:], X2[2,:], color='green')
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')

# second row = error
ax2 = fig.add_subplot(gs[1, 0])
ax2.semilogy(t, errors, color='red')
ax2.set_xlabel('t'); ax2.set_ylabel('error(t)')


"""
# Check with original code for validation
X , t = rk4_ndim(lorenz, y1_0, t0, T, dt)

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot(X[0,:], X[1,:], X[2,:])
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
"""

plt.show()

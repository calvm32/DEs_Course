import sys
sys.path.insert(0, "HW02")

import numpy as np
import matplotlib.pyplot as plt

from rk4_ndim import rk4_ndim

# ------
# Lorenz 
# ------

beta = 8/3; sigma = 10; rho = 0.2
lorenz = lambda t , x : np.array([
    sigma*(x[1] - x[0]),
    x[0]*(rho - x[2]) - x[1],
    x[0]*x[1] - beta*x[2]
])

y0 = np.array([0 , 1 , 1])    # initial value
t0 = 0.0    # initial time
T = 30      # final time
dt = 0.01   # step size

X , t = rk4_ndim(lorenz, y0, t0, T, dt)

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot(X[0,:], X[1,:], X[2,:])
ax.set_title()
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
plt.show()
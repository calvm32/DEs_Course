import sys
sys.path.insert(0, "rk4_solvers")

import numpy as np
import matplotlib.pyplot as plt
from euler_forward_error import euler_forward_error

from rk4_solvers.rk4_error import rk4_error
from rk4_solvers.rk4_ndim import rk4_ndim

# ------------------------
# Euler vs. RK4 in 2D case
# ------------------------

f = lambda t , y : -t * y **2
y_exact = lambda t : 2/(2 + t**2) 

y0 = 1.0    # initial value
t0 = 0.0    # initial time
T = 5.0     # final time
dt = 0.05   # step size

ef_errors, ef_resolutions = euler_forward_error(y_exact, f, y0, t0, T)
rk4_errors, rk4_resolutions = rk4_error(y_exact, f, y0, t0, T)

# plot the errors in a log-log plot
plt.plot(np.log(ef_resolutions) , np.log(ef_errors), "-o" , label = "Euler approximation error")
plt.plot(np.log(rk4_resolutions) , np.log(rk4_errors), "-o" , label = "RK4 approximation error")
plt.xlabel("log(resolutions)")
plt.ylabel("log(error)")
plt.legend()

plt.grid()
plt.gca().set_aspect("equal") # to judge slope
plt.show()

# -----------
# RK4 3D test
# -----------

beta = 8/3; sigma = 10; rho = 28
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
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
plt.show()
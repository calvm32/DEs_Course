import sys
sys.path.insert(0, "HW02")  # add Folder_2 path to search list <- got this solution from Google

import numpy as np
import matplotlib.pyplot as plt
import math

from rk4 import rk4
from rk4_error import rk4_error

# ----------------
# RK4 general test
# ----------------

f = lambda t , y : (y**2) - (y/3) - (t**2) # y^2-y/3-t^2

y0 = 1.0    # initial value
t0 = 0.0    # initial time
T = 5.0     # final time
dt = 0.005   # step size
M = 10**4   # threshold

y_approx , t = rk4(f, y0, t0, T, dt, M)

# plot the approximate solution
plt.plot(t , y_approx , "-o" , label = "RK4 approximation")
plt.axvline(x=math.exp(1)/2, color='r', label = "e/2")
plt.axvline(x=1.35, color='purple', label = "1.35")
plt.axvline(x=1.34, color='green', label = "1.34")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.title("M = " + str(M))
plt.show()

# -------------------
# RK4 with exact soln
# -------------------

f = lambda t , y : 0*t + y**2
y_exact = lambda t : (-1)/(t -1)

y0 = 1.0    # initial value
t0 = 0.0    # initial time
T = 5.0     # final time
dt = 0.005   # step size
M = 500    # threshold

y_approx , t = rk4(f, y0, t0, T, dt, M)
errors, resolutions = rk4_error(y_exact, f, y0, t, M)

# evaluate exact soln on interval t
y_exact_eval = np.array([],dtype = float)
for i in range(len(t)):
    y_exact_eval = np.append(y_exact_eval,y_exact(t[i]))

# plot the approximate solution
plt.subplot(1,3,1)
plt.plot(t , y_approx , "-o" , label = "RK4 approximation")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()

# plot the exact solution
plt.subplot(1,3,2)
plt.plot(t , y_exact_eval , "-o" , label = "Exact solution")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()

# plot the errors in a log-log plot
plt.subplot(1,3,3)
plt.plot(np.log(resolutions) , np.log(errors), "-o" , label = "RK4 approximation error")
plt.xlabel("log(t)")
plt.ylabel("log(error)")
plt.legend()
plt.show()
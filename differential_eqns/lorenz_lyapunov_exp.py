import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../timestep_solvers')) # from google ai search result
sys.path.append(parent_dir)

from rk4_solvers import rk4_ndim

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import solve_ivp

# -----
# setup
# -----

# random floats
a = random.uniform(-1, 1)
b = random.uniform(-1, 1)
c = random.uniform(-1, 1)

# random first initial value
y1_0 = np.array([a, b, c])

# small initial perturbation
eps = 1e-9
y2_0 = y1_0 + np.array([0, 0, eps])

# constants
t0 = 0.0                            # initial time
T = 30                              # final time
dt = 0.01                           # step size
beta = 8/3; sigma = 10; rho = 28    # parameters
N = int((T - t0) / dt) + 1          # num steps

# lorenz system
lorenz = lambda t , x : np.array([
    sigma*(x[1] - x[0]),
    x[0]*(rho - x[2]) - x[1],
    x[0]*x[1] - beta*x[2]
])

# solve and plot solution
X , t = rk4_ndim(lorenz, y1_0, t0, T, dt, 3)

##
# Now we solve for two trajectories that have a small initial separation,
# say $10^{-9}$.

ep = 1e-9
N.lbc = @(x,y,z) [x+2; y+3; z-14];
[x1,y1,z1] = N\0;         # Components of 1st trajectory
N.lbc = @(x,y,z) [x+2; y+3; z-14+ep];
[x2,y2,z2] = N\0;         # Components of 2nd trajectory

##
# Now we find the distance between trajectories using the distance formula.
# This distance, which is a function of time, is plotted using a log scale on
# the y-axis.

d = sqrt(abs(x1-x2)^2 + abs(y1-y2)^2 + abs(z1-z2)^2);
semilogy(d)
xlabel('time')
title('magnitude of separation of nearby Lorenz trajectories')

##
# The log of the distance between trajectories is well approximated by a
# straight line with positive slope, so it seems the Lorenz system has a
# positive Lyapunov exponent.

##
# Notice, however, that the positive slope only holds up for the first 25 time
# units or so. After that, the curve levels off. That is because all
# trajectories of the Lorenz system wind up in its strange attractor: since
# trajectories are bounded, they can only get so far apart.

##
# The slope of the line can be computed by finding a linear fit to the log of
# `d`. We'll only use the first 25 time units, the range where the separation
# increases exponentially.

logd = log(d{0, 25});
logd2 = polyfit(logd, 1);
slope = logd2(1) - logd2(0)

##
# And here it is for comparison to the previous plot:
hold on
x = chebfun('x', [0 dom(2)]);
semilogy(.8e-9 * exp(slope*x), 'k--')
legend('dist(traj_1, traj_2)', sprintf('exp(#1.2f x)', slope), ...
    'location', 'northwest')

##
# This approximation isn't bad at all -- the maximal Lyapunov exponent for the
# Lorenz system is known to be about $0.9056$ [3].
# To calculate it more accurately we could average over many trajectories.
# It is remarkable that this characteristic quantity of the most famous
# chaotic system is known to only a few decimal places; it is indicative of
# the difficulty in analyzing complex behavior.

## References
#
# 1. Strogatz, Steven H. _Nonlinear dynamics and chaos: with applications to
#    physics, biology and chemistry._ Perseus publishing, 2001.
#
# 2. Seydel, Rudiger. _Practical bifurcation and stability analysis._
#    Springer, 2010.
#
# 3. Viswanath, Divakar. _Lyapunov exponents from random Fibonacci sequences
#    to the Lorenz equations._ Doctoral dissertation. Cornell University,
#    1998.
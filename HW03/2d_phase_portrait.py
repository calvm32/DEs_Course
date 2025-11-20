import sys
sys.path.append("/Users/cheldt/ODEs")

from utils.ode_tools import phase_portrait
import numpy as np

x = np.linspace(0.0, 5.0, 20)  # x range
y = np.linspace(0.0, 5.0, 20)  # y range

def f(Y, t):
    x, y = Y
    return [3*x - 1.4*x*y,  # dx/dt
            -y + 0.8*x*y]  # dy/dt

phase_portrait(x, y, f)
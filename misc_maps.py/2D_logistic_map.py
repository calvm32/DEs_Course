import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math

# constants 
a = 2.8
b = 4
N_start = 0 # first iteration to plot
N_end = 10 # final iteration to plot

# ICs in (0,1)
x0 = 0.5
y0 = 0.5

x_list = [] # list of x vals
y_list = [] # list of y vals
count_list = [] # folor color mapping

sin_list = [] # for color mapping comparison
angles_list = [] # for color mapping comparison

x=x0; y=y0
for n in range(N_end):
    x = a*x*(1-y)
    y = b*y*(1-x)
    print(x)
    print(y)
    print("------")

    if n > N_start:
        x_list.append(x)
        y_list.append(y)
        count_list.append(n)

# get sine map
for n in count_list:
    x = (n-N_start)*(2 * math.pi) / (N_end - N_start) # 0 when n = N_start, 2pi when n = N_end
    angles_list.append(x)
    sin_list.append(math.sin(x))

# got this soln from google; also used in previous assignment
fig = plt.figure()
gs = GridSpec(2, 1, figure=fig, hspace=0.4) # actual layout

# sine plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(angles_list, sin_list, c=count_list, cmap='viridis')
ax1.set_title("sine map for time comparison")

# 2D logistic plot
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(x_list, y_list, c=count_list, cmap='viridis')
ax2.set_title("2D logistic map")

fig.suptitle(f"Initial condition=({x0},{y0}), a={a}, b={b}", fontsize=16, fontweight='bold') # got soln from google search ai result

plt.show()
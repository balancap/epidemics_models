import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import math
import random
import time

# Scaling
N = 10000
# Model parameters
mu = 0.1
gamma = 10
beta = 0.01
# Fix point
xstar = gamma / beta
ystar = mu / beta
# Initial values
x0 = N
y0 = N


# Model next step.
def model_next_step(t, x, y):

    t_arrival = random.expovariate(mu*x)
    t_infection = random.expovariate(beta*x*y)
    t_death = random.expovariate(gamma*y)

    if t_arrival <= t_infection and t_arrival <= t_death:
        return (t+t_arrival, x+1, y)
    elif t_infection <= t_death:
        return (t+t_infection, x-1, y+1)
    else:
        return (t+t_death, x, y-1)


# Simulation of the model
def simulate_model(_x0, _y0):
    random.seed(time.time())
    vt = np.array([0.])
    vx = np.array([_x0])
    vy = np.array([_y0])

    (t, x, y) = model_next_step(vt[-1], vx[-1], vy[-1])
    while x > 0 and y > 0 and t < 1:
        vt = np.append(vt, t)
        vx = np.append(vx, x)
        vy = np.append(vy, y)
        (t, x, y) = model_next_step(vt[-1], vx[-1], vy[-1])
        # print((t, x, y))

    return (vt, vx, vy)


# Compute conserve quantity
def f_cons_qty(x, y):
    return beta * (x+y) - gamma*math.log(x) - mu*math.log(y)
vf_cons_qty = np.vectorize(f_cons_qty)

# Make a few simulations.
nb = 1
N = 100
x0 = int(N * xstar)
y0 = int(N * ystar)

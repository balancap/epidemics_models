import numpy as np
import numba

import math
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


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
@numba.jit(nopython=True, cache=True)
def model_next_step(t, x, y):

    if x == 0:
        t_death = random.expovariate(gamma*y)
        return (t+t_death, x, y-1)
    else:
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
@numba.jit(nopython=True, cache=True)
def simulate_model(_x0, _y0, seed=0):
    random.seed(int(seed))
    vt = [0.]
    vx = [_x0]
    vy = [_y0]

    (t, x, y) = model_next_step(vt[-1], vx[-1], vy[-1])
    while x > 0 and y > 0:
        vt.append(t)
        vx.append(x)
        vy.append(y)
        (t, x, y) = model_next_step(vt[-1], vx[-1], vy[-1])

    return (np.array(vt), np.array(vx), np.array(vy))


# Compute conserve quantity
@numba.vectorize([numba.float32(numba.float32, numba.float32),
                  numba.float64(numba.float64, numba.float64)])
def f_cons_qty(x, y):
    return beta * (x+y) - gamma*math.log(x) - mu*math.log(y)
vf_cons_qty = np.vectorize(f_cons_qty)

# Make a few simulations.
nb = 1
N = 10000
x0 = int(N * xstar)
y0 = int(N * ystar)

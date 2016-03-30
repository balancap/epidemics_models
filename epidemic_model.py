# %matplotlib inline

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import matplotlib.mlab as mlab


# Model parameters
nu = 1
beta = 0.1
gamma = 10

# Plot conserve quantity of the Lotka--Volterra model

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = 10.0 * (Z2 - Z1)

# manual_locations = [(-1, -1.4), (-0.62, -0.7), (-2, 0.5),
#                     (1.7, 1.2), (2.0, 1.4), (2.4, 1.7)]

line_colours = ('BlueViolet', 'Crimson', 'ForestGreen',
        'Indigo', 'Tomato', 'Maroon')

line_widths = (1, 1.5, 2, 2.5, 3, 3.5)

plt.figure()
CS = plt.contour(X, Y, Z2, 10,                        # add 6 contour lines
                 # linewidths=line_widths,            # line widths
                 )             # line colours

plt.clabel(CS, inline=1,                            # add labels
          fontsize=10)                 # label locations
plt.title('Contour Plot - customized lines')        # title
plt.show()

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pylab
import random
import math
from scipy.stats import levy
from Generator import ExponGenerator, NormalGenerator
import random


# exponential generator test
my_gen = ExponGenerator(lamda=(1/2))
seq = [my_gen.generate_variate() for i in range(10000)]
plt.hist(seq, 100)
plt.show()

# normal generator test
my_gen = NormalGenerator(mean=0, std=1e-2)
seq = [my_gen.generate_variate() for i in range(10000)]
plt.hist(seq, 50)
plt.show()

# normal generator test
seq = [random.random() for i in range(10000)]
plt.hist(seq, 100)
plt.show()

"""
# Define parameters for the walk
dims = 1
step_n = 10000
step_set = [-1, 0, 1]
origin = np.zeros((1,dims))
# Simulate steps in 1D
step_shape = (step_n,dims)
steps = np.random.choice(a=step_set, size=step_shape)
path = np.concatenate([origin, steps]).cumsum(0)
start = path[:1]
stop = path[-1:]
# Plot the path
fig = plt.figure(figsize=(8,4),dpi=200)
ax = fig.add_subplot(111)
ax.scatter(np.arange(step_n+1), path, c='blue',alpha=0.25,s=0.05)
ax.plot(path,c='blue',alpha=0.5,lw=0.5,ls='-',)
ax.plot(0, start, c='red', marker='+')
ax.plot(step_n, stop, c='black', marker='o')
plt.title('1D Random Walk')
plt.tight_layout(pad=0)
plt.show()
"""
"""
# Define parameters for the walk
dims = 2
step_n = 10000
step_set = [-1, 0, 1]
origin = np.zeros((1,dims))
# Simulate steps in 2D
step_shape = (step_n,dims)
steps = np.random.choice(a=step_set, size=step_shape)
path = np.concatenate([origin, steps]).cumsum(0)
start = path[:1]
stop = path[-1:]
# Plot the path
fig = plt.figure(figsize=(8,8),dpi=200)
ax = fig.add_subplot(111)
ax.scatter(path[:,0], path[:,1], c='blue', alpha=0.25, s=0.05);
ax.plot(path[:,0], path[:,1], c='blue', alpha=0.5, lw=0.25, ls='-');
ax.plot(start[:,0], start[:,1],c='red', marker='+')
ax.plot(stop[:,0], stop[:,1],c='black', marker='o')
plt.title('2D Random Walk')
plt.tight_layout(pad=0)
plt.show()
"""

# Python code for 2D random walk.

"""
# defining the number of steps
n = 1000

# creating two array for containing x and y coordinate
# of size equals to the number of size and filled up with 0's
x = np.zeros(n)
y = np.zeros(n)
# x = np.linspace(0, 200, n)
# y = np.linspace(0, 200, n)
frozen_levy = levy()

# filling the coordinates with random variables
for i in range(1, n):
    angle = random.random() * 2 * math.pi
    size = frozen_levy.rvs()
    if size < 600:
        x[i] = size * math.cos(angle)
        y[i] = size * math.sin(angle)


# plotting stuff:
pylab.title("Random Walk ($n = " + str(n) + "$ steps)")
pylab.plot(np.cumsum(x), np.cumsum(y))
# pylab.savefig("rand_walk" + str(n) + ".png", bbox_inches="tight", dpi=600)
pylab.show()

"""
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 18:48:26 2020

@author: gedadav
"""
import pylab

class Plotter:
    def __init__(self, interval=1, title='no title', batch_size=10):
        self.counter = 0
        self.interval = interval
        self.batch_size = batch_size
        self.data = []
        self.title = title
        
    def update(self, data):
        self.counter += 1
        if self.counter % self.interval == self.interval - 1:
            self.data.append(data)
            
    def plot(self, x='x_axis', y='y_axis'):
        # plotting stuff:
        pylab.title(f"{self.title}\noptimization steps:{len(self.data) * self.interval * self.batch_size}")
        pylab.plot([i for i in range(len(self.data))], self.data)
        # pylab.savefig("rand_walk" + str(n) + ".png", bbox_inches="tight", dpi=600)
        pylab.show()
    
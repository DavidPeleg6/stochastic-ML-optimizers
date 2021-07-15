# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:04:21 2020

@author: gedadav
"""
import torch
from torch.optim.optimizer import Optimizer, required
import random



class ConstIntervalOptimizer(Optimizer):
    """
    Barely modified version of pytorch SGD to implement an optimizer with hops
    taken from some given distribution at constant intervals.
    """

    def __init__(self, params, generator, lr=required, interval=50):
        """
        Parameters
        ----------
        params : same as SGD
        generator : a number generator taken from some distribution
        lr : same as SGD
        interval : what is the interval to sleep between hops
        """
        self.interval = interval
        self.generator = generator
        self.counter = 0
        self.hops = 0
        defaults = dict(lr=lr)
        super(ConstIntervalOptimizer, self).__init__(params, defaults)
        
    def should_jump(self):
        return True if (self.counter % self.interval == 0) and (self.counter != 0) else False

    def step(self, lr=None):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if self.should_jump():
                    self.hops += 1
                    noise = self.generator.generate_tensor(d_p.shape)
                    p.data.sub_(d_p * group['lr'] + group['lr'] * noise)
                else:
                    p.data.sub_(d_p * group['lr'])                    
        self.counter += 1
        return loss
    
    
class VariableIntervalOptimizer(Optimizer):
    """
    Barely modified version of pytorch SGD to implement an optimizer with hops
    of constant size and interval between hops given by uniform distribution.
    """

    def __init__(self, params, lr=required, hop_size=1e-2, jump_prob=0.05):
        """
        Parameters
        ----------
        params : same as SGD
        lr : same as SGD
        hop_size : what is the size of every hop done
        jump_prob : the probability to jump
        """
        self.hop_size = hop_size
        self.jump_prob = jump_prob
        self.hops = 0
        defaults = dict(lr=lr)
        super(VariableIntervalOptimizer, self).__init__(params, defaults)

    """   TODO fix this and try again      
    def generate_noise(self, dim):
        noise = torch.rand(dim)
        temp = self.jump_prob * torch.ones(dim)
        return self.hop_size * torch.abs(-1 * torch.floor(torch.sub(noise, temp)))
    """      

    def step(self, lr=None):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                noise = torch.zeros(d_p.shape)
                if random.random() > self.jump_prob:
                    noise = self.hop_size * torch.ones(d_p.shape)
                    self.hops += 1
                p.data.sub_(d_p * group['lr'] + group['lr'] * noise)
        return loss
    
    
class ConstIntervalMemoryOptimizer(Optimizer):
    """
    a modified version of SGD that adds noise to the gradient descent after a
    constant number of optimization steps. the noise is taken as a mean of
    previous gradients.
    """
    
    def __init__(self, params, lr=required, interval=50):
        """
        Parameters
        ----------
        params : same as SGD
        lr : same as SGD
        interval : the time between jumps
        """
        self.interval = interval
        self.counter = 0
        self.hops = 0
        defaults = dict(lr=lr)
        super(ConstIntervalMemoryOptimizer, self).__init__(params, defaults)
        self.mean_grad = self.zero_means(self.param_groups[0]['params'])

    def should_jump(self):
        return (self.counter % self.interval == 0) and (self.counter != 0)
    
    def zero_means(self, params):
        means = []
        for f in params:
            # TODO change the initialization of means to this when you figure out how to do it properly
            # means.append(torch.zeros(list(f.shape) + [jump_interval]))
            means.append(torch.zeros(f.shape))
        return means

    def step(self, lr=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if self.should_jump():
            self.hops += 1
            # TODO fix this for cases with more than one param group in the optimizer
            for group in self.param_groups:
                if lr:
                    group['lr'] = lr
                # TODO find a better way to do this
                mini_counter = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    noise = (1 / self.interval) * self.mean_grad[mini_counter]
                    p.data.sub_(d_p * group['lr'] + group['lr'] * noise)
                    mini_counter += 1
                self.mean_grad = self.zero_means(group['params'])
        else:
            for group in self.param_groups:
                if lr:
                    group['lr'] = lr
                # TODO find a better way to do this
                mini_counter = 0
                for p in group['params']:
                    d_p = p.grad.data
                    self.mean_grad[mini_counter].add_(d_p)
                    p.data.sub_(d_p * group['lr']) 
                    mini_counter += 1
        self.counter += 1
        return loss
    
    
    
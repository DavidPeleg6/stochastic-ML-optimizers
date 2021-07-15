# -*- coding: utf-8 -*-
from scipy.stats import levy, expon, norm
import torch


class LevyGenerator:
    def __init__(self, scale=1):
        self.frozen_levy = levy(loc=0.001, scale=scale)

    def generate_tensor(self, shape, torch_tensor=True):
        """
        generates a tensor of random numbers taken from a levy distribution
        @:param shape = the shape of the tensor of levy variates as a list of dimensions
        @:param torch_tensor = convert the numpy array into a pytorch tensor
        @:param scale = the scale of the levy distribution the numbers are taken from (-> y = x*scale)
        """
        levy_tensor = self.frozen_levy.rvs(size=shape)
        if torch_tensor:
            levy_tensor = torch.from_numpy(levy_tensor)
        return levy_tensor
    
    def generate_variate(self):
        return self.frozen_levy.rvs()


class ExponGenerator:
    def __init__(self, lamda=10):
        self.exp = expon(scale=(1/lamda))

    def generate_tensor(self, shape, torch_tensor=True):
        """
        generates a tensor of random numbers taken from a exponential distribution
        @:param shape = the shape of the tensor of variates as a list of dimensions
        @:param torch_tensor = convert the numpy array into a pytorch tensor
        """
        exp_tensor = self.exp.rvs(size=shape)
        if torch_tensor:
            exp_tensor = torch.from_numpy(exp_tensor)
        return exp_tensor

    def generate_variate(self):
        return self.exp.rvs()


class NormalGenerator:
    def __init__(self, mean=0, std=1):
        """
        Parameters
        ----------
        mean : the mean of the normal distribution
        std : the standard deviation of the normal distribution
        """
        self.norm = norm(loc=mean, scale=std)

    def generate_tensor(self, shape, torch_tensor=True):
        """
        generates a tensor of random numbers taken from a normal distribution
        @:param shape = the shape of the tensor of variates as a list of dimensions
        @:param torch_tensor = convert the numpy array into a pytorch tensor
        """
        norm_tensor = self.norm.rvs(size=shape)
        if torch_tensor:
            norm_tensor = torch.from_numpy(norm_tensor)
        return norm_tensor

    def generate_variate(self):
        return self.norm.rvs()




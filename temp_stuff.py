# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:07:23 2020

@author: gedadav
"""
import torch

mytens = torch.tensor([[0,0,0,0],[0,0,0,0]])
nt = torch.rand(mytens.shape)
print(nt)
temp = 0.2 * torch.ones(nt.shape)
nt = torch.abs(-1 * torch.floor(torch.sub(nt, temp)))
print(nt)



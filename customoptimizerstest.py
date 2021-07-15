# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:39:51 2020

@author: gedadav
"""
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from Generator import ExponGenerator, NormalGenerator
from customoptimizers import ConstIntervalMemoryOptimizer, ConstIntervalOptimizer
from Plotter import Plotter
from datetime import datetime, date

"""
in this test run, all weights have the same learning rate for every iteration. the learning rates are taken from an
exponential distribution. once this test is done we can continue to test per-weight learning rates
"""

"""
printing time before the process starts just to log everything... TODO delete this
"""
now = datetime.now()
today = date.today()
print("Today's date:", today)
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)



#---------------------------HYPER PARAMETERS---------------------------------
alpha_0 = 1e-2
EPOCHS = 4


#-------------------------DEFINING THE ARCHITECTURE--------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # define the layer
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # define per layer actions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
    
#-----------------------------LOADING THE DATA--------------------------------
train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)


#------------------------CREATING THE NETWORK INSTANCES-----------------------
"""
#---------------------------------exponential dist networks------------------
expon_scale = [10, 10, 10]
intervals = [50, 100, 150]
net_amount = 4
nets = [{'name': None, 'net': Net(), 'generator': None, 'optimizer': None, 'plotter': None} for i in range(net_amount)]
# create all the constant interval networks
for i in range(net_amount-1):
    nets[i]['name'] = f'Constant jump interval: {intervals[i]}'
    nets[i]['plotter'] = Plotter(interval=100, title=nets[i]['name'])
    nets[i]['generator'] = ExponGenerator(lamda=expon_scale[i])
    nets[i]['optimizer'] = ConstIntervalOptimizer(nets[i]['net'].parameters(),
                                                  nets[i]['generator'], lr=alpha_0, interval=intervals[i])
    
nets[-1]['name'] = 'normal SGD'
nets[-1]['plotter'] = Plotter(interval=100, title=nets[-1]['name'])
nets[-1]['optimizer'] = SGD(nets[-1]['net'].parameters(), alpha_0)
 """
""" 
#--------------------------------------Normal dist networks------------------
net_amount = 4
mean = [0 for i in range(net_amount)]
std = [alpha_0 for i in range(net_amount)]
intervals = [1, 50, 100]
nets = [{'name': None, 'net': Net(), 'generator': None, 'optimizer': None, 'plotter': None} for i in range(net_amount)]
# create all the constant interval networks
for i in range(net_amount - 1):
    nets[i]['name'] = f'Constant jump interval\ninterval:{intervals[i]} mean:{mean[i]} std:{std[i]}'
    nets[i]['plotter'] = Plotter(interval=100, title=nets[i]['name'])
    nets[i]['generator'] = NormalGenerator(mean=mean[i], std=std[i])
    nets[i]['optimizer'] = ConstIntervalOptimizer(nets[i]['net'].parameters(),
                                                  nets[i]['generator'], lr=alpha_0, interval=intervals[i])
nets[-1]['name'] = 'normal SGD'
nets[-1]['plotter'] = Plotter(interval=100, title=nets[-1]['name'])
nets[-1]['optimizer'] = SGD(nets[-1]['net'].parameters(), alpha_0)
"""
"""
#-----------------------------------variable time interval networks-----------
net_amount = 4
hop_size = [10**-i for i in range(1, net_amount+1)]
jump_prob = [0.1*i for i in range(1, net_amount+1)]
nets = [{'name': None, 'net': Net(), 'generator': None, 'optimizer': None, 'plotter': None} for i in range(net_amount)]
# create all the constant interval networks
for i in range(net_amount - 1):
    nets[i]['name'] = f'VARIABLE JUMP INTERVAL\nsize:{hop_size[i]} jump prob:{jump_prob[i]}'
    nets[i]['plotter'] = Plotter(interval=100, title=nets[i]['name'])
    nets[i]['optimizer'] = VariableIntervalOptimizer(nets[i]['net'].parameters(), lr=alpha_0,
                                                     hop_size=hop_size[i], jump_prob=jump_prob[i])
nets[-1]['name'] = 'normal SGD'
nets[-1]['plotter'] = Plotter(interval=100, title=nets[-1]['name'])
nets[-1]['optimizer'] = SGD(nets[-1]['net'].parameters(), alpha_0)
"""

#-----------------------------------const interval with grad memory-----------
intervals = [50, 100, 150]
net_amount = 4
nets = [{'name': None, 'net': Net(), 'generator': None, 'optimizer': None, 'plotter': None} for i in range(net_amount)]
# create all the constant interval networks
for i in range(net_amount-1):
    nets[i]['name'] = f'CONSTANT INTERVAL MEMORY JUMP\ninterval:{intervals[i]}'
    nets[i]['plotter'] = Plotter(interval=100, title=nets[i]['name'])
    nets[i]['optimizer'] = ConstIntervalMemoryOptimizer(params=nets[i]['net'].parameters(),
                                                  lr=alpha_0, interval=intervals[i])
    
nets[-1]['name'] = 'normal SGD'
nets[-1]['plotter'] = Plotter(interval=100, title=nets[-1]['name'])
nets[-1]['optimizer'] = SGD(nets[-1]['net'].parameters(), alpha_0)


#------------------------TRAINING THE NETWORKS----------------------------
for i in range(EPOCHS):
    # according to the data loader, the data is sent in mini batches of 10
    for data in trainset:
        # taking a set of features and labels
        X, y = data
        # we have to reset the gradient after each iteration
        for j in range(net_amount):
            nets[j]['net'].zero_grad()
            output = nets[j]['net'](X.view(-1, 28*28))
            # a scalar value, we will use negative log likelyhood
            # if our data is a one-hot vector we use mean squared error
            loss = F.nll_loss(output, y)
            nets[j]['plotter'].update(loss)
            # we would now like to backpropogate the loss and compute the gradients
            loss.backward()
            # for now we can only use the manual learning without fancy adam
            nets[j]['optimizer'].step()
    print(f"epoch number {i+1} finished")

#-------------------------TESTING NETWORKS PERFORMANCE-----------------------
for j in range(net_amount):
    correct, total = 0, 0
    with torch.no_grad():
        for data in testset:
            # split into label and sample and run them through the network
            X, y = data
            output = nets[j]['net'](X.view(-1, 28 * 28))
            # print(output)
            # iterate over all the outputs and compare the network results to the data results
            for idx, ans in enumerate(output):
                # ans is the vector of probablities for the i'th example and idx is the index of the current example in the batch
                # print(torch.argmax(ans), idx)
                if torch.argmax(ans) == y[idx]:
                    correct += 1
                total += 1
    # claculate accuracy up to 3 digits after the dot
    print(f"{nets[j]['name']} accuracy: ", round(correct / total, 3))
    print(f"hops:{nets[j]['optimizer'].hops}") if j < net_amount - 1 else print('no hops')
    nets[j]['plotter'].plot()


# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:19:36 2020

@author: gedadav
"""

import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from Generator import ExponGenerator
import torch.optim as optim
from Plotter import Plotter

"""
in this test run, we will allow a jump every 50 steps. the jump will be calculated as the mean 
of the previous 50 steps for each parameter. other than that, the optimizer is a simple SGD
"""

# testing a manual gradient calculation
# hyper parameters
# we implemented it in a way such that the default step size is small and the variate lr is much larger
alpha_0 = 1e-2
expon_scale = 1
jump_interval = 50


# a function to calculate whether we should jump
def should_jump(x):
    return True if (x % jump_interval == 0) and (x != 0) else False

def zero_means(param_list):
    means = []
    for f in param_list:
        # TODO change the initialization of means to this when you figure out how to do it properly
        # means.append(torch.zeros(list(f.shape) + [jump_interval]))
        means.append(torch.zeros_like(f))
    return means


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



# then we test it with per-weight lr
net = Net()
# initialize the counter to figure out when to jump
jump_counter = 0
# for anlaysis later
total_jumps = 0
# initialize exponential distribution generator
# TODO try it with and without the exponential generator and with different scales
generator = ExponGenerator(scale=expon_scale)
# generate a list of tensors the size of the parameters
means = zero_means(net.parameters())

print("training")
loss_plot = Plotter(interval=100)
EPOCHS = 50
for i in range(EPOCHS):
    # according to the data loader, the data is sent in mini batches of 10
    for data in trainset:
        # taking a set of features and labels
        X, y = data
        # we have to reset the gradient after each iteration
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        # a scalar value, we will use negative log likelyhood
        # if our data is a one-hot vector we use mean squared error
        loss = F.nll_loss(output, y)
        loss_plot.update(loss)
        # we would now like to backpropogate the loss and compute the gradients
        loss.backward()
        # a counter for the mean iterations
        mini_counter = 0   
        if should_jump(jump_counter):
            for f in net.parameters():
                # TODO maybe delete this
                means[mini_counter].add_(f.grad.data)
                f.data.sub_(generator.generate_tensor(f.shape) * 
                            ((1/jump_interval) * means[mini_counter]))
                mini_counter += 1
            means = zero_means(net.parameters())
            jump_counter = 0
            total_jumps += 1
            continue
        
        for f in net.parameters():
            f.data.sub_(alpha_0 * f.grad.data)
            # add the current value to mean
            means[mini_counter].add_(f.grad.data)
            mini_counter += 1
        # increment the counter every iteration
        jump_counter += 1
        

print("testing")
correct, total = 0, 0
with torch.no_grad():
    for data in testset:
        # split into label and sample and run them through the network
        X, y = data
        output = net(X.view(-1, 28 * 28))
        # print(output)
        # iterate over all the outputs and compare the network results to the data results
        for idx, ans in enumerate(output):
            # ans is the vector of probablities for the i'th example and idx is the index of the current example in the batch
            # print(torch.argmax(ans), idx)
            if torch.argmax(ans) == y[idx]:
                correct += 1
            total += 1
# claculate accuracy up to 3 digits after the dot
print("per weight lr accuracy: ", round(correct / total, 3))
loss_plot.plot(title="per-weight learning rate")





# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 12:52:28 2020

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
from numpy import log

"""
this version of the multiplicative noise combines three approaches. Meaning 3 neural networks are trained in a different manner
the importance of this test is that not only the data is taken from the same set but the minibatches are also the same
1. a regular SGD
2. a SGD with jumps taken from exponential distribution after 50 steps
3. a SGD with jumps taken from exponential distribution with time also taken from exponential distribution
    the rule for the exponential jump is if x < 0.0513 which is similar to 20% of the time
"""

"""
printing time before the process starts just to log everything... TODO delete this
"""
from datetime import datetime, date

now = datetime.now()
today = date.today()
print("Today's date:", today)
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


# testing a manual gradient calculation
# hyper parameters
# we implemented it in a way such that the default step size is small and the variate lr is much larger
alpha_0 = 1e-2
expon_scale = (1/5)
jump_prob = 0.05
jump_interval = 50
spec_jump_lim = (1/expon_scale) * log(1/(1-jump_prob))
EPOCHS = 25



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



# creating 3 networks
# TODO change this to be a list of networks so you can add more variations
# and change this to a new class that inherits from the normal network and changes the iteration methods
normal_net = Net()
const_jump_net = Net()
special_jump_net = Net()
# initialize the counter to figure out when to jump
jump_counter = 0
special_jump_counter = 0
# for anlaysis later
total_const_jumps = 0
total_special_jumps = 0
# initialize exponential distribution generator
# TODO try it with and without the exponential generator and with different scales
const_jump_generator = ExponGenerator(lamda=expon_scale)
special_jump_generator = ExponGenerator(lamda=expon_scale)
# generate a list of tensors the size of the parameters
const_jump_means = zero_means(const_jump_net.parameters())
special_jump_means = zero_means(special_jump_net.parameters())
optimizer = optim.SGD(normal_net.parameters(), lr=alpha_0)

print("training")
normal_loss_plot = Plotter(interval=100)
const_loss_plot = Plotter(interval=100)
special_loss_plot = Plotter(interval=100)
for i in range(EPOCHS):
    # according to the data loader, the data is sent in mini batches of 10
    for data in trainset:
        # taking a set of features and labels
        X, y = data
        # we have to reset the gradient after each iteration
        normal_net.zero_grad()
        const_jump_net.zero_grad()
        special_jump_net.zero_grad()
        
        normal_output = normal_net(X.view(-1, 28*28))
        const_output = const_jump_net(X.view(-1, 28*28))
        special_output = special_jump_net(X.view(-1, 28*28))

        # a scalar value, we will use negative log likelyhood
        # if our data is a one-hot vector we use mean squared error
        normal_loss = F.nll_loss(normal_output, y)
        const_loss = F.nll_loss(const_output, y)
        special_loss = F.nll_loss(special_output, y)

        normal_loss_plot.update(normal_loss)
        const_loss_plot.update(const_loss)
        special_loss_plot.update(special_loss)

        # we would now like to backpropogate the loss and compute the gradients
        normal_loss.backward()
        const_loss.backward()
        special_loss.backward()

        # check whether you should make a normal jump
        const_jump = should_jump(jump_counter)
        if const_jump:
            # a counter for the mean iterations
            mini_counter = 0   
            for f in const_jump_net.parameters():
                # TODO maybe delete this
                const_jump_means[mini_counter].add_(f.grad.data)
                f.data.sub_(const_jump_generator.generate_tensor(f.shape) * 
                            ((1/jump_interval) * const_jump_means[mini_counter]))
                mini_counter += 1
            const_jump_means = zero_means(const_jump_net.parameters())
            jump_counter = 0
            total_const_jumps += 1
          
        # check whether you should make a special jump (the number generator returned a number greater than the limit)
        special_jump = True if special_jump_generator.generate_variate() <= spec_jump_lim else False
        if special_jump and special_jump_counter != 0:
            # a counter for the mean iterations
            mini_counter = 0   
            for f in special_jump_net.parameters():
                # TODO maybe delete this
                special_jump_means[mini_counter].add_(f.grad.data)
                f.data.sub_(special_jump_generator.generate_tensor(f.shape) * 
                            ((1/special_jump_counter) * special_jump_means[mini_counter]))
                mini_counter += 1
            special_jump_means = zero_means(special_jump_net.parameters())
            special_jump_counter = 0
            total_special_jumps += 1
        
        optimizer.step()

        if not const_jump:
            mini_counter = 0
            for f in const_jump_net.parameters():
                f.data.sub_(alpha_0 * f.grad.data)
                # add the current value to mean
                const_jump_means[mini_counter].add_(f.grad.data)
                mini_counter += 1
            # increment the counter every iteration
            jump_counter += 1
            
        if not special_jump or special_jump_counter == 0:
            mini_counter = 0
            for f in special_jump_net.parameters():
                f.data.sub_(alpha_0 * f.grad.data)
                # add the current value to mean
                special_jump_means[mini_counter].add_(f.grad.data)
                mini_counter += 1
            # increment the counter every iteration
            special_jump_counter += 1
        
print("testing")
correct, total = [0, 0, 0], [0, 0, 0]
with torch.no_grad():
    for data in testset:
        # split into label and sample and run them through the network
        X, y = data
        normal_output = normal_net(X.view(-1, 28 * 28))
        const_output = const_jump_net(X.view(-1, 28 * 28))
        special_output = special_jump_net(X.view(-1, 28 * 28))

        # iterate over all the outputs and compare the network results to the data results
        for idx, ans in enumerate(normal_output):
            # ans is the vector of probablities for the i'th example and idx is the index of the current example in the batch
            # print(torch.argmax(ans), idx)
            if torch.argmax(ans) == y[idx]:
                correct[0] += 1
            total[0] += 1
            
        # iterate over all the outputs and compare the network results to the data results
        for idx, ans in enumerate(const_output):
            # ans is the vector of probablities for the i'th example and idx is the index of the current example in the batch
            # print(torch.argmax(ans), idx)
            if torch.argmax(ans) == y[idx]:
                correct[1] += 1
            total[1] += 1
        
        # iterate over all the outputs and compare the network results to the data results
        for idx, ans in enumerate(special_output):
            # ans is the vector of probablities for the i'th example and idx is the index of the current example in the batch
            # print(torch.argmax(ans), idx)
            if torch.argmax(ans) == y[idx]:
                correct[2] += 1
            total[2] += 1

# claculate accuracy up to 3 digits after the dot
print("normal: ", round(correct[0] / total[0], 3))
normal_loss_plot.plot(title="normal SGD")

print(f"const jump interval of: {jump_interval}\n accuracy for lamda {expon_scale}: {round(correct[1] / total[1], 3)}")
const_loss_plot.plot(title="constant jump SGD")

print(f"special jump with probability: {jump_prob}\naccuracy for lamda {expon_scale}: {round(correct[2] / total[2], 3)}")
special_loss_plot.plot(title="special jump SGD")


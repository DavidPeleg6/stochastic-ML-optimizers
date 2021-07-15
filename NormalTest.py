# -*- coding: utf-8 -*-
import torch
import torchvision
import pylab
import numpy as np
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Plotter import Plotter


alpha_0 = 1e-2

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


net = Net()
# print(net)

print("training")
# parameters are essentially the weights of the network, so net.parameters returns an iterable of the weights
optimizer = optim.SGD(net.parameters(), lr=alpha_0)
# yields better results but I wish to see the impact better
# optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_plot = Plotter(interval=100)
EPOCHS = 3
for i in range(EPOCHS):
    # according to the data loader, the data is sent in mini batches of 10
    iter_count = 0
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
        # we would now like to backpropogate the loss and computing the gradients
        loss.backward()
        # change the learning rate
        # now we make the optimizer change the weights
        optimizer.step()
    #print(f"loss: {loss} in epoch: {i}")

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
print("normal SGD accuracy: ", round(correct / total, 3))

loss_plot.plot(title='SGD normal')




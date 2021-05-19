import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Transforms images from [0,255] to [0,1] range.
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0], std=[1])])

# Load the set of training images.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# Load the set of test images.
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# Let's plot some of the images to see what we're dealing with.
def plot_images(imgs):
    for i in range(imgs.size()[0]):
        npimg = imgs.numpy()[i,0,:,:]
        plt.imshow(npimg, cmap='gray')
        plt.ion()
        plt.show()
        plt.pause(.05)

data = iter(testloader)
images, labels = data.next()
print(labels)
plot_images(images)

import torch.nn as nn
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.dense_linear = nn.Sequential(nn.Linear(3 * 32 * 32, 10))
    def forward(self, x):
        x = x.view(-1,3*32*32)
        x = self.dense_linear(x)
        return x
model = LogisticRegression()

# This library contains implementations of a number of useful optimization algorithms.
import torch.optim as optim

# The cross entropy loss is already implemented in Pytorch.
criterion = nn.CrossEntropyLoss()

# The stochastic gradient descent algorithm with a step size of 0.1.
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Write a loop to train the model using the given optimizer and loss functions.
for i in range(20):
    for data in trainloader:
    # extract the images and labels.
        inputs, labels = data
    
    # This must be called to zero out the accumulated gradients.
        optimizer.zero_grad()
    
    # Calculate the predictions made based on the model.
        outputs = model(inputs)
    
    # Evaluate the loss function based on the model predictions.
        loss = criterion(outputs, labels)
    
    # Calculate the gradient of the parameters with respect to the loss.
        loss.backward()
    
    # Take a optimization step.
        optimizer.step()
    
    print('Completed epoch %d' % i)
print('Completed training')

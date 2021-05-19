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
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=4)

# Load the set of test images.
testset = torchvision.datasets.MNIST(root='./data', train=False,
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
# plot_images(images)

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
model = LeNet5()

# This library contains implementations of a number of useful optimization algorithms.
import torch.optim as optim

weight1 = model.conv1.weight
weight2 = model.conv2.weight
weight3 = model.fc1.weight
weight4 = model.fc2.weight
weight5 = model.fc3.weight
alpha = 0.1

loss1 = torch.max(torch.sum(torch.abs(weight1), dim = [1,2,3]))
loss2 = torch.max(torch.sum(torch.abs(weight2), dim = [1,2,3]))
loss3 = torch.max(torch.sum(torch.abs(weight3), dim = 1))
loss4 = torch.max(torch.sum(torch.abs(weight4), dim = 1))
loss5 = torch.max(torch.sum(torch.abs(weight5), dim = 1))

regularization = loss1 + loss2 + loss3 + loss4 + loss5


# The cross entropy loss is already implemented in Pytorch.
criterion = nn.CrossEntropyLoss()

# The stochastic gradient descent algorithm with a step size of 0.1.
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Write a loop to train the model using the given optimizer and loss functions.
for i in range(30):
    for data in trainloader:
    # extract the images and labels.
        inputs, labels = data
    
    # This must be called to zero out the accumulated gradients.
        optimizer.zero_grad()
    
    # Calculate the predictions made based on the model.
        outputs = model(inputs)
    
    # Evaluate the loss function based on the model predictions.
        loss = criterion(outputs, labels) + alpha * regularization
    
    # Calculate the gradient of the parameters with respect to the loss.
        loss.backward()
    
    # Take a optimization step.
        optimizer.step()
    
    print('Completed epoch %d' % i)
print('Completed training')

# Calculate the total number of test samples and the number of correctly 
# classified test samples
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = model(images)
  # Take the most likely label as the predicted label.
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Out of %d samples, the model correctly classified %d' % (total, correct))


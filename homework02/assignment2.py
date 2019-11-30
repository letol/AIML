"""**Import libraries**"""

import os
import logging
import git

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn

import torchvision
from torchvision import transforms
from torchvision.models import alexnet

from PIL import Image
from tqdm import tqdm

# %%
"""**Set Arguments**"""

DEVICE = 'cuda'  # 'cuda' or 'cpu'

NUM_CLASSES = 102  # 101 + 1: There is an extra Background class that should be removed

BATCH_SIZE = 256
# Higher batch sizes allows for larger learning rates. An empirical heuristic suggests that, when changing
# the batch size, learning rate should change by the same factor to have comparable results

LR = 1e-3  # The initial Learning Rate
MOMENTUM = 0.9  # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default

NUM_EPOCHS = 30  # Total number of training epochs (iterations over dataset)
STEP_SIZE = 20  # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.1  # Multiplicative factor for learning rate step-down

LOG_FREQUENCY = 10

# %%
"""**Define Data Preprocessing**"""

# Define transforms for training phase
train_transform = transforms.Compose([transforms.Resize(256),  # Resizes short size of the PIL image to 256
                                      transforms.CenterCrop(224),  # Crops a central square patch of the image 224
                                      # because torchvision's AlexNet needs a 224x224 input! Remember this when
                                      # applying different transformations, otherwise you get an error
                                      transforms.ToTensor(),  # Turn PIL Image to torch.Tensor
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      # Normalizes tensor with mean and standard deviation
                                      ])
# Define transforms for the evaluation phase
eval_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])

# %%
"""**Prepare Dataset**"""

# Clone github repository with data
if not os.path.isdir('./Homework2-Caltech101'):
    git.cmd.Git().clone("https://github.com/MachineLearning2020/Homework2-Caltech101.git")

DATA_DIR = 'Homework2-Caltech101/101_ObjectCategories'

# Prepare Pytorch train/test Datasets
train_dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=train_transform)
test_dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=eval_transform)

train_indexes = [idx for idx in range(len(train_dataset)) if idx % 5]
test_indexes = [idx for idx in range(len(test_dataset)) if not idx % 5]

train_dataset = Subset(train_dataset, train_indexes)
test_dataset = Subset(test_dataset, test_indexes)

# Check dataset sizes
print('Train Dataset: {}'.format(len(train_dataset)))
print('Test Dataset: {}'.format(len(test_dataset)))

# %%
"""**Prepare Dataloaders**"""

# Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and
# shuffling)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

#%%
"""**Prepare Network**"""

net = alexnet()  # Loading AlexNet model

# AlexNet has 1000 output neurons, corresponding to the 1000 ImageNet's classes
# We need 101 outputs for Caltech-101
net.classifier[6] = nn.Linear(4096, NUM_CLASSES)  # nn.Linear in pytorch is a fully connected layer
# The convolutional layer is nn.Conv2d

# We just changed the last layer of AlexNet with a new fully connected layer with 101 outputs
# It is mandatory to study torchvision.models.alexnet source code

# %%
"""**Prepare Training**"""

# Define loss function
criterion = nn.CrossEntropyLoss()  # for classification, we use Cross Entropy

# Choose parameters to optimize
# To access a different set of parameters, you have to access submodules of AlexNet
# (nn.Module objects, like AlexNet, implement the Composite Pattern)
# e.g.: parameters of the fully connected layers: net.classifier.parameters()
# e.g.: parameters of the convolutional layers: look at alexnet's source code ;) 
parameters_to_optimize = net.parameters()  # In this case we optimize over all the parameters of AlexNet

# Define optimizer
# An optimizer updates the weights based on loss
# We use SGD with momentum
optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# Define scheduler
# A scheduler dynamically changes learning rate
# The most common schedule is the step(-down), which multiplies learning rate by gamma every STEP_SIZE epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# %%
"""**Train**"""

# By default, everything is loaded to cpu
net = net.to(DEVICE)  # this will bring the network to GPU if DEVICE is cuda

cudnn.benchmark  # Calling this optimizes runtime

current_step = 0
# Start iterating over the epochs
for epoch in range(NUM_EPOCHS):
    print('Starting epoch {}/{}, LR = {}'.format(epoch + 1, NUM_EPOCHS, scheduler.get_lr()))

    # Iterate over the dataset
    for images, labels in train_dataloader:
        # Bring data over the device of choice
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        net.train()  # Sets module in training mode

        # PyTorch, by default, accumulates gradients after each backward pass
        # We need to manually set the gradients to zero before starting a new iteration
        optimizer.zero_grad()  # Zero-ing the gradients

        # Forward pass to the network
        outputs = net(images)

        # Compute loss based on output and ground truth
        loss = criterion(outputs, labels)

        # Log loss
        if current_step % LOG_FREQUENCY == 0:
            print('Step {}, Loss {}'.format(current_step, loss.item()))

        # Compute gradients for each layer and update weights
        loss.backward()  # backward pass: computes gradients
        optimizer.step()  # update weights based on accumulated gradients

        current_step += 1

    # Step the scheduler
    scheduler.step()

# %%
"""**Test**"""

net = net.to(DEVICE)  # this will bring the network to GPU if DEVICE is cuda
net.train(False)  # Set Network to evaluation mode

running_corrects = 0
for images, labels in tqdm(test_dataloader):
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    # Forward Pass
    outputs = net(images)

    # Get predictions
    _, preds = torch.max(outputs.data, 1)

    # Update Corrects
    running_corrects += torch.sum(preds == labels.data).data.item()

# Calculate Accuracy
accuracy = running_corrects / float(len(test_dataset))

print('Test Accuracy: {}'.format(accuracy))

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
from torchvision import transforms as tr
from torchvision.models import alexnet

from PIL import Image
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import caltech_dataset
from caltech_dataset import train_valid_split

"""**Define Functions**"""


def evaluate(network, dataset, dataloader, multiple_crops=False):
    '''
    The evaluate method returns the accuracy of the given model calculated on the test_dataset provided,
    using the test_dataloader to load the items
    '''
    network = network.to(DEVICE)  # this will bring the network to GPU if DEVICE is cuda
    network.train(False)  # Set Network to evaluation mode

    running_corrects = 0
    for imgs, lbls in tqdm(dataloader):
        imgs = imgs.to(DEVICE)
        lbls = lbls.to(DEVICE)

        if multiple_crops:
            bs, ncrops, c, h, w = imgs.size()
            out = network(imgs.view(-1, c, h, w))  # fuse batch size and ncrops
            out = out.view(bs, ncrops, -1).mean(1)  # avg over crops
        else:
            out = network(imgs)  # Forward Pass

        # Get predictions
        _, preds = torch.max(out.data, 1)

        # Update Corrects
        running_corrects += torch.sum(preds == lbls.data).data.item()

    # Calculate Accuracy
    acc = running_corrects / float(len(dataset))
    return acc


def select_layers(network, layer_class):
    '''
    The select_layers method returns an iterator over the parameters of selected layers
    Args:
        network (nn.Module): original network module
        layer_class (type): class of layers to be selected
    Returns:
        Iterator over parameters
    '''
    for layer in network.modules():
        if isinstance(layer, layer_class):
            for parameter in layer.parameters():
                yield parameter


# %%
"""**Set Arguments**"""

DEVICE = 'cuda'  # 'cuda' or 'cpu'

NUM_CLASSES = 101

BATCH_SIZE = 256
# Higher batch sizes allows for larger learning rates. An empirical heuristic suggests that, when changing
# the batch size, learning rate should change by the same factor to have comparable results

LR = 1e-2  # The initial Learning Rate
MOMENTUM = 0.9  # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default

NUM_EPOCHS = 30  # Total number of training epochs (iterations over dataset)
STEP_SIZE = 20  # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.1  # Multiplicative factor for learning rate step-down

LOG_FREQUENCY = 10

# %%
"""**Define Data Preprocessing**"""

# Define transforms for training phase
train_transform = tr.Compose([tr.Resize(256),  # Resizes short size of the PIL image to 256
                              tr.CenterCrop(224),  # Crops a central square patch of the image 224
                              # because torchvision's AlexNet needs a 224x224 input! Remember this when
                              # applying different transformations, otherwise you get an error
                              # /======================================================================================\
                              # 4.A: Data Augmentation
                              # ----------------------------------------------------------------------------------------
                              # transforms.RandomHorizontalFlip(),
                              # transforms.RandomPerspective(distortion_scale=0.2),
                              # transforms.RandomRotation(degrees=10),
                              # ----------------------------------------------------------------------------------------
                              tr.RandomChoice([tr.RandomHorizontalFlip(),
                                               tr.RandomPerspective(distortion_scale=0.2),
                                               tr.RandomRotation(degrees=10)]),
                              # \======================================================================================/
                              tr.ToTensor(),  # Turn PIL Image to torch.Tensor
                              # /======================================================================================\
                              # Normalizes tensor with mean and standard deviation
                              # ----------------------------------------------------------------------------------------
                              # Till 3.A:
                              # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              # ----------------------------------------------------------------------------------------
                              # From 3.B on:
                              tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                              # \======================================================================================/
                              ])
# Define transforms for the evaluation phase
eval_transform = tr.Compose([tr.Resize(256),
                             # /=======================================================================================\
                             # 4.A: Data Augmentation
                             # -----------------------------------------------------------------------------------------
                             tr.Compose([tr.TenCrop(224),
                                         tr.Lambda(
                                             lambda crops: torch.stack(
                                                 [tr.Compose([tr.ToTensor(),
                                                              tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                              ])(crop) for crop in crops]
                                             ))
                                         ]),
                             # -----------------------------------------------------------------------------------------
                             # transforms.CenterCrop(224),
                             # transforms.ToTensor(),
                             # /=======================================================================================\
                             # Till 3.A:
                             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             # -----------------------------------------------------------------------------------------
                             # From 3.B on:
                             # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                             # \=======================================================================================/
                             # \=======================================================================================/
                             ])

# %%
"""**Prepare Dataset**"""

# Clone github repository with data
if not os.path.isdir('./Homework2-Caltech101'):
    git.cmd.Git().clone("https://github.com/MachineLearning2020/Homework2-Caltech101.git")

DATA_DIR = 'Homework2-Caltech101/101_ObjectCategories'

# Prepare Pytorch train/test Datasets
train_dataset = caltech_dataset.Caltech(DATA_DIR, split='train', transform=train_transform)
test_dataset = caltech_dataset.Caltech(DATA_DIR, split='test', transform=eval_transform)

print('train.txt Dataset: {}'.format(len(train_dataset)))
print('test.txt Dataset: {}\n'.format(len(test_dataset)))

train_idx, valid_idx = train_valid_split(train_dataset, len(train_dataset.targets))

valid_dataset = Subset(train_dataset, valid_idx)
train_dataset = Subset(train_dataset, train_idx)

# Check dataset sizes
print('Train Dataset: {}'.format(len(train_dataset)))
print('Validation Dataset: {}'.format(len(valid_dataset)))
print('Test Dataset: {}\n'.format(len(test_dataset)))

# %%
"""**Prepare Dataloaders**"""

# Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and
# shuffling)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

#%%
"""**Prepare Network**"""

# /====================================================================================================================\
# Till 2.C:
# net = alexnet()  # Loading AlexNet model
# ----------------------------------------------------------------------------------------------------------------------
# From 3.A on:
net = alexnet(pretrained=True)
# \====================================================================================================================/

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
# /====================================================================================================================\
# Till 3.C and from 4.A on: In this case we optimize over all the parameters of AlexNet
# parameters_to_optimize = net.parameters()
# ----------------------------------------------------------------------------------------------------------------------
# 3.D: In this case we optimize only the fully connected layers
parameters_to_optimize = net.classifier.parameters()
# ----------------------------------------------------------------------------------------------------------------------
# 3.E: In this case we optimize only the convolutional layers
# parameters_to_optimize = select_layers(net, nn.Conv2d)
# \====================================================================================================================/

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
accuracies = []
losses = []
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
        losses.append(loss)

        # Log loss
        if current_step % LOG_FREQUENCY == 0:
            print('Step {}, Loss {}'.format(current_step, loss.item()))

        # Compute gradients for each layer and update weights
        loss.backward()  # backward pass: computes gradients
        optimizer.step()  # update weights based on accumulated gradients

        current_step += 1

    accuracy = evaluate(net, valid_dataset, valid_dataloader)
    accuracies.append(accuracy)
    print('Validation Accuracy at epoch {}/{}: {}'.format(epoch + 1, NUM_EPOCHS, accuracy))

    # Step the scheduler
    scheduler.step()

plt.figure()
plt.title('Accuracies with LR={}, STEP_SIZE={}'.format(LR, STEP_SIZE))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.plot(np.arange(1, NUM_EPOCHS+1, 1.0), accuracies)
plt.show()

steps_per_epoch = len(losses) / NUM_EPOCHS
xticks_step = 5    # in epochs
plt.figure()
plt.title('Losses with LR={}, STEP_SIZE={}'.format(LR, STEP_SIZE))
plt.xlabel('Epoch')
plt.xticks(np.arange(-steps_per_epoch, len(losses) + 1, ((len(losses) + steps_per_epoch) / NUM_EPOCHS) * xticks_step),
           np.arange(0, NUM_EPOCHS+1, xticks_step))
plt.ylabel('Loss')
plt.ylim(0, 5)
plt.plot(np.arange(1, len(losses)+1, 1.0), losses)
plt.show()

# %%

"""**Test**"""

# test_accuracy = evaluate(net, test_dataset, test_dataloader)
test_accuracy = evaluate(net, test_dataset, test_dataloader, multiple_crops=True)

print('Test Accuracy: {}'.format(test_accuracy))

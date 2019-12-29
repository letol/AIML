"""**Import libraries**"""

import os
import git

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.backends import cudnn

from torchvision import transforms as tr
from torchvision.models import alexnet
from torchvision.datasets import ImageFolder

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt


"""**Define Functions**"""


def evaluate(network, dataset, dataloader, multiple_crops=False):
    '''
    The evaluate function returns the accuracy of the given model calculated on the test_dataset provided,
    using the test_dataloader to load the items
    Args:
        network (nn.Module): network to be evaluated
        dataset (VisionDataset): test dataset
        dataloader (DataLoader): test dataset dataloader
        multiple_crops (bool): Set to True if test dataloader includes multiple crops transfomation (eg.: TenCrop)
    Returns:
        accuracy (float)
    '''
    network = network.to(DEVICE)  # this will bring the network to GPU if DEVICE is cuda
    network.train(False)  # Set Network to evaluation mode

    running_corrects = 0
    for imgs, lbls in tqdm(dataloader):
        imgs = imgs.to(DEVICE)
        lbls = lbls.to(DEVICE)

        if multiple_crops:
            bs, ncrops, c, h, w = imgs.size()
            out = network(imgs.view(-1, c, h, w))  # Fuse batch size and ncrops
            out = out.view(bs, ncrops, -1).mean(1)  # Avg over crops
        else:
            out = network(imgs)  # Forward Pass

        # Get predictions
        _, preds = torch.max(out.data, 1)

        # Update Corrects
        running_corrects += torch.sum(preds == lbls.data).data.item()

    # Calculate Accuracy
    acc = running_corrects / float(len(dataset))
    return acc


# %%
"""**Set Arguments**"""

DEVICE = 'cuda'  # 'cuda' or 'cpu'

MODEL_DIR = './models'
MODEL_NAME = 'model.pth'

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
                              tr.ToTensor(),  # Turn PIL Image to torch.Tensor
                              # Normalizes tensor with mean and standard deviation
                              tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                              ])
# Define transforms for the evaluation phase
eval_transform = tr.Compose([tr.Resize(256),
                             tr.CenterCrop(224),
                             tr.ToTensor(),
                             tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                             ])

# %%
"""**Prepare Dataset**"""

# Clone github repository with data
if not os.path.isdir('./Homework3-PACS'):
    git.cmd.Git().clone("https://github.com/MachineLearning2020/Homework3-PACS.git")

P_DIR = 'Homework3-PACS/PACS/photo'
A_DIR = 'Homework3-PACS/PACS/art_painting'
C_DIR = 'Homework3-PACS/PACS/cartoon'
S_DIR = 'Homework3-PACS/PACS/sketch'

# Prepare Pytorch train/test Datasets
photo_dataset = ImageFolder(P_DIR, transform=train_transform)
art_dataset = ImageFolder(A_DIR, transform=train_transform)
cartoon_dataset = ImageFolder(C_DIR, transform=eval_transform)
sketch_dataset = ImageFolder(S_DIR, transform=eval_transform)

# Check dataset sizes
print('Photo Dataset: {}'.format(len(photo_dataset)))
print('Art Painting Dataset: {}'.format(len(art_dataset)))
print('Cartoon Dataset: {}'.format(len(cartoon_dataset)))
print('Sketch Dataset: {}\n'.format(len(sketch_dataset)))

# %%
"""**Prepare Dataloaders**"""

# Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and
# shuffling)
photo_dataloader = DataLoader(photo_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
art_dataloader = DataLoader(art_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
cartoon_dataloader = DataLoader(cartoon_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
sketch_dataloader = DataLoader(sketch_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

#%%
"""**Prepare Network**"""

net = alexnet(pretrained=True)

net.classifier[6] = nn.Linear(4096, NUM_CLASSES)

# %%
"""**Prepare Training**"""

# Define loss function
criterion = nn.CrossEntropyLoss()  # for classification, we use Cross Entropy

# Choose parameters to optimize
parameters_to_optimize = net.parameters()

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

        # Add loss to history
        losses.append(loss)

        # Log loss
        if current_step % LOG_FREQUENCY == 0:
            print('Step {}, Loss {}'.format(current_step, loss.item()))

        # Compute gradients for each layer and update weights
        loss.backward()  # backward pass: computes gradients
        optimizer.step()  # update weights based on accumulated gradients

        current_step += 1

    # Evaluate the model on validation set
    accuracy = evaluate(net, valid_dataset, valid_dataloader)

    # Add accuracy to history
    accuracies.append(accuracy)

    # Log accuracy
    print('Validation Accuracy at epoch {}/{}: {}'.format(epoch + 1, NUM_EPOCHS, accuracy))

    # Step the scheduler
    scheduler.step()

# Plot accuracy history
plt.figure()
plt.title('Accuracies with LR={}, STEP_SIZE={}'.format(LR, STEP_SIZE))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.plot(np.arange(1, NUM_EPOCHS+1, 1.0), accuracies)
plt.show()

# Plot loss history
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

# Save model
if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)
torch.save(net, os.path.join(MODEL_DIR, MODEL_NAME))

# %%
"""**Test**"""

# Load model
net = torch.load(os.path.join(MODEL_DIR, MODEL_NAME))

# Evaluate the model on test set
test_accuracy = evaluate(net, test_dataset, test_dataloader)

print('Test Accuracy: {}'.format(test_accuracy))

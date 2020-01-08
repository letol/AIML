"""**Import libraries**"""

import os
import git

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.backends import cudnn

from torchvision import transforms as tr
from DANN import dann
from torchvision.datasets import ImageFolder
from sklearn.model_selection import ParameterGrid

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math

#%%
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


def get_model_name(params, source_dataset, target_dataset, adapt):
    model_name = 'model-' + os.path.basename(source_dataset.root) + '_' + os.path.basename(target_dataset.root) + \
                 '-lr_' + str(10 ** params['lr_exp']) + \
                 '-step_size_' + str(params['step_size']) + \
                 '-epochs_' + str(params['num_epochs'])
    if adapt:
        model_name = model_name + \
                     '-alpha_' + str(10 ** params['alpha_exp']) + \
                     '-adapt'
    else:
        model_name = model_name + \
                     '-no_adapt'

    return model_name.translate(str.maketrans('.', '_')) + '.pt'


# %%
"""**Set Arguments**"""

DEVICE = 'cuda'  # 'cuda' or 'cpu'

NUM_CLASSES = 7

BATCH_SIZE = 64
# Higher batch sizes allows for larger learning rates. An empirical heuristic suggests that, when changing
# the batch size, learning rate should change by the same factor to have comparable results

LOG_FREQUENCY = 10

MODEL_DIR = './models_64'

#%%
"""**Define Data Preprocessing**"""

# Define transforms for training phase
train_transform = tr.Compose([tr.Resize(224),  # Resizes short size of the PIL image to 256
                              # tr.CenterCrop(224),  # Crops a central square patch of the image 224
                              # because torchvision's AlexNet needs a 224x224 input! Remember this when
                              # applying different transformations, otherwise you get an error
                              tr.ToTensor(),  # Turn PIL Image to torch.Tensor
                              # Normalizes tensor with mean and standard deviation
                              tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                              ])
# Define transforms for the evaluation phase
eval_transform = tr.Compose([tr.Resize(224),
                             # tr.CenterCrop(224),
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


# %%
"""**Prepare Training**"""


def training(model, source_dataset, source_dataloader, target_dataset, target_dataloader, params, adaptation=False, evalu=True, cache=True):
    if cache:
        model_name = get_model_name(params, source_dataset, target_dataset, adaptation)
        # If already computed, load model
        if os.path.exists(os.path.join(MODEL_DIR, model_name)):
            print("Pre-trained model found!")
            model = torch.load(os.path.join(MODEL_DIR, model_name))
            return model['net'], model['losses'], model['accuracies']

    if params['step_size'] > params['num_epochs']:
        print("step_size > num_epochs! => next")
        return None, None, None

    # Define loss function
    criterion = nn.CrossEntropyLoss()  # for classification, we use Cross Entropy

    # Choose parameters to optimize
    parameters_to_optimize = model.parameters()

    # Define optimizer
    # An optimizer updates the weights based on loss
    # We use SGD with momentum
    optimizer = optim.SGD(parameters_to_optimize, lr=(10**params['lr_exp']), momentum=params['momentum'], weight_decay=params['weight_decay'])

    # Define scheduler
    # A scheduler dynamically changes learning rate
    # The most common schedule is the step(-down), which multiplies learning rate by gamma every STEP_SIZE epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])

    # By default, everything is loaded to cpu
    net = model.to(DEVICE)  # this will bring the network to GPU if DEVICE is cuda

    cudnn.benchmark  # Calling this optimizes runtime

    current_step = 0
    source_accuracies = []
    target_accuracies = []
    ly_losses = []
    ld_source_losses = []
    ld_target_losses = []

    # Start iterating over the epochs
    for epoch in range(params['num_epochs']):
        print('Starting epoch {}/{}, LR = {}'.format(epoch + 1, params['num_epochs'], scheduler.get_lr()))

        if adaptation:
            target_dataloader_iterator = iter(target_dataloader)

        # Iterate over the dataset
        for images, labels in source_dataloader:
            # Bring data over the device of choice
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            zeros = torch.zeros_like(labels)
            ones = torch.ones_like(labels)

            net.train()  # Sets module in training mode

            # PyTorch, by default, accumulates gradients after each backward pass
            # We need to manually set the gradients to zero before starting a new iteration
            optimizer.zero_grad()  # Zero-ing the gradients

            # Forward pass source images to the Gy branch of the network
            outputs = net(images)

            # Compute Ly loss based on output and ground truth
            ly_loss = criterion(outputs, labels)
            if math.isnan(ly_loss):
                print("NaN Loss! => next")
                return None, None, None

            # Add Ly loss to history
            ly_losses.append(ly_loss)

            # Compute gradients for each layer and update weights
            ly_loss.backward()  # backward pass: computes gradients

            # Log Ly loss
            if current_step % LOG_FREQUENCY == 0:
                print('Step {}, Gy Loss {}'.format(current_step, ly_loss.item()))

            if adaptation:
                # Forward pass source images to the Gd branch of the network
                outputs = net(images, alpha=10**params['alpha_exp'])

                # Compute source images Ld loss based on output and ground truth (all zeros)
                ld_source_loss = criterion(outputs, zeros)
                if math.isnan(ld_source_loss):
                    print("NaN Loss! => next")
                    return None, None, None

                # Add source images Ld loss to history
                ld_source_losses.append(ld_source_loss)

                # Compute gradients for each layer and update weights
                ld_source_loss.backward()  # backward pass: computes gradients

                # Get target dataset batch
                images, _ = next(target_dataloader_iterator)
                images = images.to(DEVICE)

                # Forward pass target images to the Gd branch of the network
                outputs = net(images, alpha=10**params['alpha_exp'])

                # Compute target images Ld loss based on output and ground truth (all ones)
                ld_target_loss = criterion(outputs, ones)
                if math.isnan(ld_target_loss):
                    print("NaN Loss! => next")
                    return None, None, None

                # Add target images Ld loss to history
                ld_target_losses.append(ld_target_loss)

                # Compute gradients for each layer and update weights
                ld_target_loss.backward()  # backward pass: computes gradients

                # Log Gd losses
                if current_step % LOG_FREQUENCY == 0:
                    print('Step {}, Gd Source Loss {}'.format(current_step, ld_source_loss.item()))
                    print('Step {}, Gd Target Loss {}'.format(current_step, ld_target_loss.item()))

            optimizer.step()  # update weights based on accumulated gradients

            current_step += 1

        # Evaluate the model on validation set
        if evalu:
            source_accuracy = evaluate(net, source_dataset, source_dataloader)
            target_accuracy = evaluate(net, target_dataset, target_dataloader)

            # Add accuracy to history
            source_accuracies.append(source_accuracy)
            target_accuracies.append(target_accuracy)

            # Log accuracy
            print('Source/Target Accuracy at epoch {}/{}: {}/{}'.format(epoch + 1, params['num_epochs'], source_accuracy, target_accuracy))

        # Step the scheduler
        scheduler.step()

    losses = {'ly': ly_losses, 'ld_source': ld_source_losses, 'ld_target': ld_target_losses}
    accuracies = {'source': source_accuracies, 'target': target_accuracies}

    # Save Model
    if cache:
        if not os.path.isdir(MODEL_DIR):
            os.mkdir(MODEL_DIR)
        torch.save({'net': net, 'losses': losses, 'accuracies': accuracies}, os.path.join(MODEL_DIR, model_name))

    return net, losses, accuracies


#%%
"""**Prepare Grid Search**"""


def grid_search(param_grid, adapt):
    param_list = list(ParameterGrid(param_grid))

    best_params = None
    best_acc_avg = 0
    best_losses = {'cartoon': None, 'sketch': None}
    best_accuracies = {'cartoon': None, 'sketch': None}

    for it, params in enumerate(param_list, start=1):
        print("params {}/{}: {}".format(it, len(param_list), params))

        print("Starting Photo to Cartoon training")
        model = dann(pretrained=True)
        model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
        model, cartoon_losses, cartoon_accuracies = training(model,
                                                             photo_dataset, photo_dataloader,
                                                             cartoon_dataset, cartoon_dataloader,
                                                             params,
                                                             adaptation=adapt)
        if model is None:
            continue

        print("Starting Photo to Sketch training")
        model = dann(pretrained=True)
        model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
        model, sketch_losses, sketch_accuracies = training(model,
                                                           photo_dataset, photo_dataloader,
                                                           sketch_dataset, sketch_dataloader,
                                                           params,
                                                           adaptation=adapt)
        if model is None:
            continue

        print("Target Accuracy Photo-Cartoon: {}".format(cartoon_accuracies['target'][-1]))
        print("Target Accuracy Photo-Sketch: {}".format(sketch_accuracies['target'][-1]))

        acc_avg = float((cartoon_accuracies['target'][-1] + sketch_accuracies['target'][-1]) / 2)

        print("Average Accuracy: {}".format(acc_avg))

        if acc_avg > best_acc_avg:
            print("New best params found!")
            best_params = params
            best_acc_avg = acc_avg
            best_losses['cartoon'] = cartoon_losses
            best_accuracies['cartoon'] = cartoon_accuracies
            best_losses['sketch'] = sketch_losses
            best_accuracies['sketch'] = sketch_accuracies

    return best_params, best_losses, best_accuracies


#%%
"""**Grid Search without Adaptation**"""

param_grid = {'lr_exp': [-1, -2, -3],  # The initial Learning Rate
              'momentum': [0.9],  # Hyperparameter for SGD, keep this at 0.9 when using SGD
              'weight_decay': [5e-5],  # Regularization, you can keep this at the default
              'num_epochs': [10, 20, 30],   # Total number of training epochs (iterations over dataset)
              'step_size': [10, 20, 30],  # How many epochs before decreasing learning rate (if using a step-down policy)
              'gamma': [0.1]}  # Multiplicative factor for learning rate step-down

best_params, valid_losses, valid_accuracies = grid_search(param_grid, adapt=False)

print("Best params: {}".format(best_params))

#%%
"""**Train without Adaptation**"""

model = dann(pretrained=True)

model.classifier[6] = nn.Linear(4096, NUM_CLASSES)

trained_model_no_adapt, losses, _ = training(model,
                                             photo_dataset, photo_dataloader,
                                             art_dataset, art_dataloader,
                                             best_params,
                                             adaptation=False,
                                             evalu=False)

#%%
"""**Some plots**"""

#Plot Photo-Cartoon accuracy history
plt.figure()
plt.title('Photo-Cartoon Accuracies with LR={}, STEP_SIZE={}'.format(10**best_params['lr_exp'], best_params['step_size']))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.plot(np.arange(1, best_params['num_epochs']+1, 1.0), valid_accuracies['cartoon']['source'], 'b', label='source')
plt.plot(np.arange(1, best_params['num_epochs']+1, 1.0), valid_accuracies['cartoon']['target'], 'g', label='target')
plt.show()

#Plot Photo-Sketch accuracy history
plt.figure()
plt.title('Photo-Sketch Accuracies with LR={}, STEP_SIZE={}'.format(10**best_params['lr_exp'], best_params['step_size']))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.plot(np.arange(1, best_params['num_epochs']+1, 1.0), valid_accuracies['sketch']['source'], 'b', label='source')
plt.plot(np.arange(1, best_params['num_epochs']+1, 1.0), valid_accuracies['sketch']['target'], 'g', label='target')
plt.legend()
plt.show()

# Plot Photo-Cartoon loss history
steps_per_epoch = len(valid_losses['cartoon']['ly']) / best_params['num_epochs']
xticks_step = 5    # in epochs
plt.figure()
plt.title('Photo-Cartoon Losses with LR={}, STEP_SIZE={}'.format(10**best_params['lr_exp'], best_params['step_size']))
plt.xlabel('Epoch')
plt.xticks(np.arange(-steps_per_epoch, len(valid_losses['cartoon']['ly']) + 1, ((len(valid_losses['cartoon']['ly']) + steps_per_epoch) / best_params['num_epochs']) * xticks_step),
           np.arange(0, best_params['num_epochs']+1, xticks_step))
plt.ylabel('Loss')
plt.plot(np.arange(1, len(valid_losses['cartoon']['ly']) + 1, 1.0), valid_losses['cartoon']['ly'], 'b', label='Ly')
plt.legend()
plt.show()

# Plot Photo-Sketch loss history
steps_per_epoch = len(valid_losses['sketch']['ly']) / best_params['num_epochs']
xticks_step = 5    # in epochs
plt.figure()
plt.title('Photo-Sketch Losses with LR={}, STEP_SIZE={}'.format(10**best_params['lr_exp'], best_params['step_size']))
plt.xlabel('Epoch')
plt.xticks(np.arange(-steps_per_epoch, len(valid_losses['sketch']['ly']) + 1, ((len(valid_losses['sketch']['ly']) + steps_per_epoch) / best_params['num_epochs']) * xticks_step),
           np.arange(0, best_params['num_epochs']+1, xticks_step))
plt.ylabel('Loss')
plt.plot(np.arange(1, len(valid_losses['sketch']['ly']) + 1, 1.0), valid_losses['sketch']['ly'], 'b', label='Ly')
plt.legend()
plt.show()

# Plot Photo-Art loss history
steps_per_epoch = len(losses['ly']) / best_params['num_epochs']
xticks_step = 5    # in epochs
plt.figure()
plt.title('Photo-Art Losses with LR={}, STEP_SIZE={}'.format(10**best_params['lr_exp'], best_params['step_size']))
plt.xlabel('Epoch')
plt.xticks(np.arange(-steps_per_epoch, len(losses['ly']) + 1, ((len(losses['ly']) + steps_per_epoch) / best_params['num_epochs']) * xticks_step),
           np.arange(0, best_params['num_epochs']+1, xticks_step))
plt.ylabel('Loss')
plt.plot(np.arange(1, len(losses['ly']) + 1, 1.0), losses['ly'], 'b', label='Ly')
plt.legend()
plt.show()

#%%
"""**Grid Search with Adaptation**"""

param_grid = {'lr_exp': [-1, -2, -3],  # The initial Learning Rate
              'momentum': [0.9],  # Hyperparameter for SGD, keep this at 0.9 when using SGD
              'weight_decay': [5e-5],  # Regularization, you can keep this at the default
              'alpha_exp': [-1, -2, -3, -4, -5],  # Weight of reversed backpropagation
              'num_epochs': [10, 20, 30],   # Total number of training epochs (iterations over dataset)
              'step_size': [10, 20, 30],  # How many epochs before decreasing learning rate (if using a step-down policy)
              'gamma': [0.1]}  # Multiplicative factor for learning rate step-down

best_params, valid_losses, valid_accuracies = grid_search(param_grid, adapt=True)

print("Best params: {}".format(best_params))

#%%
"""**Train with Adaptation**"""

model = dann(pretrained=True)

model.classifier[6] = nn.Linear(4096, NUM_CLASSES)

trained_model_adapt, losses, _ = training(model,
                                          photo_dataset, photo_dataloader,
                                          art_dataset, art_dataloader,
                                          best_params,
                                          adaptation=True,
                                          evalu=False)

#%%
"""**Some other plots**"""

#Plot Photo-Cartoon accuracy history
plt.figure()
plt.title('Photo-Cartoon Accuracies with LR={}, STEP_SIZE={}, ALPHA={}'.format(10**best_params['lr_exp'], best_params['step_size'], 10**best_params['alpha_exp']))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.plot(np.arange(1, best_params['num_epochs']+1, 1.0), valid_accuracies['cartoon']['source'], 'b', label='source')
plt.plot(np.arange(1, best_params['num_epochs']+1, 1.0), valid_accuracies['cartoon']['target'], 'g', label='target')
plt.show()

#Plot Photo-Sketch accuracy history
plt.figure()
plt.title('Photo-Sketch Accuracies with LR={}, STEP_SIZE={}, ALPHA={}'.format(10**best_params['lr_exp'], best_params['step_size'], 10**best_params['alpha_exp']))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.plot(np.arange(1, best_params['num_epochs']+1, 1.0), valid_accuracies['sketch']['source'], 'b', label='source')
plt.plot(np.arange(1, best_params['num_epochs']+1, 1.0), valid_accuracies['sketch']['target'], 'g', label='target')
plt.legend()
plt.show()

# Plot Photo-Cartoon loss history
steps_per_epoch = len(valid_losses['cartoon']['ly']) / best_params['num_epochs']
xticks_step = 5    # in epochs
plt.figure()
plt.title('Photo-Cartoon Losses with LR={}, STEP_SIZE={}, ALPHA={}'.format(10**best_params['lr_exp'], best_params['step_size'], 10**best_params['alpha_exp']))
plt.xlabel('Epoch')
plt.xticks(np.arange(-steps_per_epoch, len(valid_losses['cartoon']['ly']) + 1, ((len(valid_losses['cartoon']['ly']) + steps_per_epoch) / best_params['num_epochs']) * xticks_step),
           np.arange(0, best_params['num_epochs']+1, xticks_step))
plt.ylabel('Loss')
plt.plot(np.arange(1, len(valid_losses['cartoon']['ly']) + 1, 1.0), valid_losses['cartoon']['ly'], 'b', label='Ly')
plt.plot(np.arange(1, len(valid_losses['cartoon']['ld_source']) +1, 1.0), valid_losses['cartoon']['ld_source'], 'g', label='Ld Source')
plt.plot(np.arange(1, len(valid_losses['cartoon']['ld_target']) +1, 1.0), valid_losses['cartoon']['ld_target'], 'r', label='Ld Target')
plt.legend()
plt.show()

# Plot Photo-Sketch loss history
steps_per_epoch = len(valid_losses['sketch']['ly']) / best_params['num_epochs']
xticks_step = 5    # in epochs
plt.figure()
plt.title('Photo-Sketch Losses with LR={}, STEP_SIZE={}, ALPHA={}'.format(10**best_params['lr_exp'], best_params['step_size'], 10**best_params['alpha_exp']))
plt.xlabel('Epoch')
plt.xticks(np.arange(-steps_per_epoch, len(valid_losses['sketch']['ly']) + 1, ((len(valid_losses['sketch']['ly']) + steps_per_epoch) / best_params['num_epochs']) * xticks_step),
           np.arange(0, best_params['num_epochs']+1, xticks_step))
plt.ylabel('Loss')
plt.plot(np.arange(1, len(valid_losses['sketch']['ly']) + 1, 1.0), valid_losses['sketch']['ly'], 'b', label='Ly')
plt.plot(np.arange(1, len(valid_losses['sketch']['ld_source']) +1, 1.0), valid_losses['sketch']['ld_source'], 'g', label='Ld Source')
plt.plot(np.arange(1, len(valid_losses['sketch']['ld_target']) +1, 1.0), valid_losses['sketch']['ld_target'], 'r', label='Ld Target')
plt.legend()
plt.show()

# Plot Photo-Art loss history
steps_per_epoch = len(losses['ly']) / best_params['num_epochs']
xticks_step = 5    # in epochs
plt.figure()
plt.title('Photo-Art Losses with LR={}, STEP_SIZE={}, ALPHA={}'.format(10**best_params['lr_exp'], best_params['step_size'], 10**best_params['alpha_exp']))
plt.xlabel('Epoch')
plt.xticks(np.arange(-steps_per_epoch, len(losses['ly']) + 1, ((len(losses['ly']) + steps_per_epoch) / best_params['num_epochs']) * xticks_step),
           np.arange(0, best_params['num_epochs']+1, xticks_step))
plt.ylabel('Loss')
plt.plot(np.arange(1, len(losses['ly']) + 1, 1.0), losses['ly'], 'b', label='Ly')
plt.plot(np.arange(1, len(losses['ld_source']) +1, 1.0), losses['ld_source'], 'g', label='Ld Source')
plt.plot(np.arange(1, len(losses['ld_target']) +1, 1.0), losses['ld_target'], 'r', label='Ld Target')
plt.legend()
plt.show()

#%%
"""**Test**"""

# Evaluate the model on test set
test_accuracy_no_adapt = evaluate(trained_model_no_adapt, art_dataset, art_dataloader)
test_accuracy_adapt = evaluate(trained_model_adapt, art_dataset, art_dataloader)

print('Test Accuracy without Adaptation: {}'.format(test_accuracy_no_adapt))
print('Test Accuracy with Adaptation: {}'.format(test_accuracy_adapt))

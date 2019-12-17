from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import random


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def train_valid_split(dataset, num_targets):
    '''
    The train_valid_split method splits the training set in training and validation sets and returns them.
    It aims for half samples of each class in training set and the other half in validation set.
    Args:
        dataset (VisionDataset): dataset to be splitted
        num_targets (int): number of targets present in given dataset

    Returns:
        tuple : (train_idx, valid_idx)
    '''
    classes = [[] for i in range(num_targets)]
    [classes[sample[1]].append(idx) for idx, sample in enumerate(dataset)]

    train_idx = []
    valid_idx = []

    for c in classes:
        random.shuffle(c)
        split = int(len(c)/2)
        [train_idx.append(idx) for idx in c[split:]]
        [valid_idx.append(idx) for idx in c[:split]]

    return train_idx, valid_idx


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split  # This defines the split you are going to use
        # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

        if split == 'train':
            path = os.path.join(os.path.dirname(root), 'train.txt')
        else:
            path = os.path.join(os.path.dirname(root), 'test.txt')

        split_file = open(path, 'r')

        image_paths = split_file.readlines()

        self.targets = []
        self.images = []

        for subpath in image_paths:
            subpath = subpath.replace('\n', '')
            folder = os.path.dirname(subpath)

            if not folder.startswith('BACKGROUND'):
                if folder not in self.targets:
                    self.targets.append(folder)

                label = self.targets.index(folder)
                path = os.path.join(root, subpath)

                self.images.append((pil_loader(path), label))

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.images[index]  # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.images)  # Provide a way to get the length (number of elements) of the dataset
        return length

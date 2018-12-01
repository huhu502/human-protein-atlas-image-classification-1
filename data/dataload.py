import numpy as np
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import SubsetRandomSampler
from .dataset import MultiBandMultiLabelDataset


def get_dateloaders(csv_file, 
                    data_dir,
                    train_transform,
                    valid_transform,
                    batch_size = 32,
                    random_seed = 30,
                    valid_size=0.1,
                    shuffle=True,
                    num_workers=5):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    train_dataset = MultiBandMultiLabelDataset(csv_file=csv_file, root_dir=data_dir, transform=train_transform)
    valid_dataset = MultiBandMultiLabelDataset(csv_file=csv_file, root_dir=data_dir, transform=valid_transform)

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_size * dataset_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                sampler=valid_sampler, num_workers=num_workers)
    dataloaders = {
        'train': train_loader,
        'val': valid_loader
    }
    return dataloaders
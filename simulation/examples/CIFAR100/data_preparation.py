"""Functions for dataset download and processing."""
import random
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from flwr.common.logger import log
from logging import DEBUG, INFO


# Define a custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def load_datasets(  # pylint: disable=too-many-arguments
    num_clients: int,
    val_ratio: float = 0.2,
    batch_size: Optional[int] = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create the dataloaders to be fed into the model.

    Parameters
    ----------
    config: DictConfig
        Parameterises the dataset partitioning process
    num_clients : int
        The number of clients that hold a part of the data
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    batch_size : int, optional
        The size of the batches to be fed into the model, by default 32
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for validation, the DataLoader
        for testing.
    """
    datasets, testset = _partition_data(num_clients)
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) / (1 / val_ratio))
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(dataset, lengths)
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    return trainloaders, valloaders, DataLoader(testset, batch_size=batch_size)


def _download_data(num_clients) -> Tuple[Dataset, Dataset]:
    
    # CIFAR-100 Data Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download CIFAR-100 Dataset
    train_data = datasets.CIFAR100(root='/home/ccl/Desktop/FedOps/examples/simulation/dataset/cifar100', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR100(root='/home/ccl/Desktop/FedOps/examples/simulation/dataset/cifar100', train=False, download=True, transform=transform)

    return train_data, test_data

# Set Client Data Partition
def _partition_data(
    num_clients,
) -> Tuple[List[Dataset], Dataset]:

    train_data, test_data = _download_data(num_clients)
    
    # Calculate the total number of samples in the train dataset
    total_size = len(train_data)

    # Randomly generate split sizes for each client that sum up to total_size
    sizes = _generate_random_sizes(total_size, num_clients)

    # Randomly split the train_data into num_clients parts
    train_data_partition = random_split(train_data, sizes)

    return train_data_partition, test_data


def _generate_random_sizes(total_size: int, num_clients: int) -> List[int]:
    """Generate random sizes that sum up to total_size."""
    sizes = []
    for _ in range(num_clients - 1):
        size = random.randint(1, total_size - sum(sizes) - (num_clients - len(sizes)))
        sizes.append(size)
    sizes.append(total_size - sum(sizes))  # Add the remaining size to the last client
    return sizes
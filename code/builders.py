"""
High-level to build data and models given experiment arguments
"""

import models
import data

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def build_dataloaders(args, random_state=None):
    """
    Build dataloaders according to experiment args

    Parameters
    ----------
    args : argparse.Namespace
        Experiment arguments
    random_state : np.random.RandomState, optional (default: None)
        Optional random state for reproducibility in train/val/test split
    """
    # Load raw data
    raw_data = data.load_raw_data(args.data_file)

    # Train/val/test split
    datas = data.train_val_test_split(raw_data, random_state=random_state)
    dataloaders = {}

    # Wrap in DataLoaders
    for split, dataset in datas.items():
        dataloaders[split] = DataLoader(dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        pin_memory=args.pin_memory)
    return dataloaders


def build_model(args):
    """
    Build the model, optimizer, and loss according to experiment args

    Parameters
    ----------
    args : argparse.Namespace
        Experiment arguments

    Returns
    -------
    model : torch.nn.Module
        The model to be trained
    optimizer : torch.optim.Optimizer
        The optimizer
    loss : torch.nn.Module
        The loss function
    """
    # Build the model according to teh given arguments
    model = models.LogisticRegression(init_m=args.init_m, init_b=args.init_b)

    # Adam is a good default optimizer choice
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Your loss depends on the problem
    loss = nn.BCEWithLogitsLoss()

    if args.cuda:
        model = model.cuda()
        loss = loss.cuda()

    return model, optimizer, loss

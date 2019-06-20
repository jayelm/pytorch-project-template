"""
Main script for training models
"""

# Python system/generic modules
import os
from collections import defaultdict
import contextlib

# Specialized packages
import numpy as np

# Torch
import torch

# This module
import builders
import io_util
import util

# Logging
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def init_metrics():
    """
    Initialize the metrics for this training run. This is a defaultdict, so
    metrics not specified here can just be appended to/assigned to during
    training.

    Returns
    -------
    metrics : `collections.defaultdict`
        All training metrics
    """
    metrics = defaultdict(list)
    metrics['best_acc'] = 0.0
    metrics['best_loss'] = float('inf')
    metrics['best_epoch'] = 0
    return metrics


def compute_average_metrics(meters):
    """
    Compute averages from meters. Handle tensors vs floats (always return a
    float)

    Parameters
    ----------
    meters : Dict[str, util.AverageMeter]
        Dict of average meters, whose averages may be of type ``float`` or ``torch.Tensor``

    Returns
    -------
    metrics : Dict[str, float]
        Average value of each metric
    """
    metrics = {m: vs.avg for m, vs in meters.items()}
    metrics = {m: v if isinstance(v, float) else v.item() for m, v in metrics.items()}
    return metrics


def run(split, epoch, model, optimizer, loss, dataloaders, args,
        random_state=None):
    """
    Run the model for a single epoch.

    Parameters
    ----------
    split : ``str``
        The dataloader split to use. Also determines model behavior if e.g.
        ``split == 'train'`` then model will be in train mode/optimizer will be
        run.
    epoch : ``int``
        current epoch
    model : ``torch.nn.Module``
        the model you are training/evaling
    optimizer : ``torch.nn.optim.Optimizer``
        the optimizer
    loss : ``torch.nn.loss``
        the loss function
    dataloaders : ``dict[str, torch.utils.data.DataLoader]``
        Dictionary of dataloaders whose keys are the names of the ``split``s
        and whose values are the corresponding dataloaders
    args : ``argparse.Namespace``
        Arguments for this experiment run
    random_state : ``np.random.RandomState``
        The numpy random state in case anything stochastic happens during the
        run

    Returns
    -------
    metrics : ``dict[str, float]``
        Metrics from this run; keys are statistics and values are their average
        values across the batches
    """
    training = split == 'train'
    dataloader = dataloaders[split]
    if training:
        model.train()
        context = contextlib.suppress  # Null context (does nothing)
    else:
        model.eval()
        context = torch.no_grad  # Do not evaluate gradients for efficiency

    # Initialize your average meters to keep track of the epoch's running average
    measures = ['loss', 'acc']
    meters = {m: util.AverageMeter() for m in measures}

    with context():
        for batch_i, (x, y) in enumerate(dataloader):
            batch_size = x.shape[0]
            if args.cuda:
                x = x.cuda()

            # Refresh the optimizer
            if training:
                optimizer.zero_grad()

            # Forward pass
            scores = model(x)

            # Evaluate loss and accuracy
            this_loss = loss(scores, y)
            # Usually the metric we actually care about is not the loss function we're optimizing...
            # here contains logic for actually evaluating the end metric (in
            # this case classification accuracy)
            yhat = (scores > 0).float()
            this_acc = (yhat == y).float().mean().item()

            if training:
                # SGD step
                this_loss.backward()
                optimizer.step()

            meters['loss'].update(this_loss, batch_size)
            meters['acc'].update(this_acc, batch_size)

            # Log your progress through the epoch
            if training and batch_i % args.log_interval == 0:
                logging.info('Epoch {}\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_i * batch_size, len(dataloader.dataset),
                    100 * batch_i / len(dataloader), meters['loss'].avg))

    metrics = compute_average_metrics(meters)
    logging.info('Epoch {}\t{} {}'.format(
        epoch, split.upper(), ' '.join('{}: {:.4f}'.format(m, v) for m, v in metrics.items())
    ))
    return metrics


if __name__ == '__main__':
    # Description is this script's docstring
    args = io_util.parse_args('train', desc=__doc__)

    # Check if resumable;
    resumable = args.resume and io_util.is_resumable(args.exp_dir)
    # Make the exp directory if exists.
    # NOTE: Current script does not care if an experiment directory already
    # exists and will overwrite existing files. Be careful with this! You may
    # want to change this default behavior.
    os.makedirs(args.exp_dir, exist_ok=True)
    if not resumable:
        io_util.save_args(args, args.exp_dir)

    # Always seed for reproducibility!
    random = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    # Use the builders to quickly build your data, model, optimizers
    dataloaders = builders.build_dataloaders(args, random_state=random)
    model, optimizer, loss = builders.build_model(args)

    # If resume, load metrics; otherwise init metrics
    if resumable:
        io_util.restore_checkpoint(model, optimizer, args.exp_dir)

        metrics = io_util.load_metrics(args.exp_dir)
        start_epoch = metrics['current_epoch'] + 1
        logging.info("Resuming from epoch {}".format(metrics['current_epoch']))
    else:
        metrics = init_metrics()
        start_epoch = 1

    for epoch in range(start_epoch, args.epochs + 1):
        # Run your train and validation steps.
        train_metrics = run('train', epoch, model, optimizer, loss, dataloaders, args, random_state=random)
        val_metrics = run('val', epoch, model, optimizer, loss, dataloaders, args, random_state=random)

        # Update your metrics, prepending the split name.
        for metric, value in train_metrics.items():
            metrics['train_{}'.format(metric)].append(value)
        for metric, value in val_metrics.items():
            metrics['val_{}'.format(metric)].append(value)
        metrics['current_epoch'] = epoch

        # Use validation accuracy to choose the best model. If it's the best,
        # update the best metrics.
        is_best = val_metrics['acc'] > metrics['best_acc']
        if is_best:
            metrics['best_acc'] = val_metrics['acc']
            metrics['best_loss'] = val_metrics['loss']
            metrics['best_epoch'] = epoch

        # Save model and optimizer state and current epoch.
        state_dict = {
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'epoch': epoch
        }
        io_util.save_checkpoint(state_dict, is_best, args.exp_dir)
        if (epoch % args.save_interval == 0):
            io_util.save_checkpoint(state_dict, False, args.exp_dir,
                                 filename='{}.pth'.format(epoch))

        # Save metrics at each epoch (overriding existing file)
        io_util.save_metrics(metrics, args.exp_dir)

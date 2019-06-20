"""
This file handles argument parsing and loading/serialization for all scripts in
this repository.
"""


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import subprocess
import json
import os
import shutil

import torch


def current_git_hash():
    """
    Get the hash of the latest commit in this repository. Does not account for unstaged changes.

    Returns
    -------
    git_hash : ``str``, optional
        The string corresponding to the current git hash if known, else ``None`` if something failed.
    """
    try:
        git_hash = subprocess.check_output(['git', 'describe', '--always']).strip().decode('utf-8')
        return git_hash
    except:
        return None


def is_resumable(exp_dir, metrics_file='metrics.json', checkpoint_file='checkpoint.pth'):
    """
    Check if an experiment directory is resumable.

    An experiment directory if resumable if (1) it contains a metrics file and
    (2) a recent checkpoint.

    Parameters
    ----------
    exp_dir : ``str``
        Experiment directory to check

    Returns
    -------
    is_resumable : ``bool``
        True if checkpoint is resumable, else False
    """
    return (
        os.path.exists(os.path.join(exp_dir, metrics_file)) and
        os.path.exists(os.path.join(exp_dir, checkpoint_file)))


def save_checkpoint(state, is_best, exp_dir, filename='checkpoint.pth',
                    best_filename='model_best.pth'):
    """
    Save a checkpoint

    Parameters
    ----------
    state : ``dict``
        State dictionary consisting of model state to save (generally with
        string keys and state_dict values).
    is_best : ``bool``
        Is the model the best one encountered during this training run?
        If so, copy over the model (normally saved to ``filename`` to
        ``best_filename``.)
    exp_dir : ``str``
        Experiment directory to save to.
    filename : ``str``, optional (default: 'checkpoint.pth')
        Model name to save
    best_filename : ``str``, optional (default: 'model_best.pth')
        Model name to save if this model is the best version
    """
    torch.save(state, os.path.join(exp_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(exp_dir, filename),
                        os.path.join(exp_dir, 'model_best.pth'))


def load_checkpoint(exp_dir, filename='checkpoint.pth', device=None):
    """
    Load a checkpoint.

    Parameters
    ----------
    exp_dir : ``str``
        Experiment directory to load from.
    filename : ``str``, optional (default: 'checkpoint.pth')
        Name of the checkpoint to load
    device : ``str``, optional (default: None)
        Which device to load to; defaults to device models were originally
        specified with.

    Returns
    -------
    state : ``dict``
        State dictionary of checkpoint.
    """
    return torch.load(os.path.join(exp_dir, filename), map_location=device)


def restore_checkpoint(model, optimizer, exp_dir, filename='checkpoint.pth',
                       device=None):
    """
    Restore a model checkpoint into the given model and optimizer in-place.

    Parameters
    ----------
    model : ``torch.nn.Module``
        Model to restore.
    optimizer : ``torch.nn.optim.Optimizer``
        Optimizer to restore.
    exp_dir : ``str``
        Experiment directory to load from.
    filename : ``str``, optional (default: 'checkpoint.pth')
        Name of the checkpoint to load
    device : ``str``, optional (default: None)
        Which device to load to; defaults to device models were originally
        specified with.
    """
    ckpt = load_checkpoint(exp_dir, filename=filename, device=device)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])


def restore_args(args, exp_dir, filename='args.json'):
    """
    Load arguments from an experiment directory and populate the given
    namespace in place. Does NOT override arguments already in the namespace.

    This wraps ``load_args`` and is generally used over that function in other
    scripts, since what you want is to fill in an already-existing
    `argparse.Namespace` object.

    Parameters
    ----------
    args : ``argparse.Namespace``
        Namespace to populate with loaded arguments (will NOT replace existing
        arguments!). Namespace will be modified in place.
    exp_dir : ``str``
        Folder to load args from
    filename : ``str``, optional (default: 'args.json')
        Name of argument file
    """
    exp_args = load_args(exp_dir, filename=filename)
    for arg, val in exp_args.items():
        if not arg in args:
            args.__setattr__(arg, val)


def load_args(exp_dir, filename='args.json'):
    """
    Load arguments from experiment directory into ``dict`` format. This is used
    mainly to build a model given a configuration that was supplied at training
    time and saved into an experiment directory.

    The ``dict`` format here is crucial; this means that loading arguments from
    a folder will not behave like a normal ``argparse.Namespace`` object. Since
    most scripts will require generating some sort of (incomplete)
    ``argparse.Namespace``, it's recommended to use the ``restore_args``
    function in this module.

    Parameters
    ----------
    exp_dir : ``str``
        Folder to load args from
    filename : ``str``, optional (default: 'args.json')
        Name of argument file

    Returns
    -------
    args : ``dict``
        Dictionary of loaded arguments.
    """
    with open(os.path.join(exp_dir, filename), 'r') as f:
        args = json.load(f)
        # Delete git hash
        if 'git_hash' in args:
            del args['git_hash']


def save_args(args, exp_dir, filename='args.json'):
    """
    Save arguments in the experiment directory. This is REALLY IMPORTANT for
    reproducibility, so you know exactly what configuration of arguments
    resulted in what experiment result! As a bonus, this function also saves
    the current git hash so you know exactly which version of the code produced
    your result (that is, as long as you don't run with unstaged changes).

    Parameters
    ----------
    args : ``argparse.Namespace``
        Arguments to save
    exp_dir : ``str``
        Folder to save args to
    filename : ``str``, optional (default: 'args.json')
        Name of argument file
    """
    args_dict = vars(args)
    args_dict['git_hash'] = current_git_hash()
    with open(os.path.join(exp_dir, filename), 'w') as f:
        json.dump(args_dict, f, indent=4, separators=(',', ': '), sort_keys=True)


def load_metrics(exp_dir, filename='metrics.json'):
    """
    Load metrics from the given exp directory..

    Parameters
    ----------
    exp_dir : ``str``
        Folder to load metrics from
    filename : ``str``, optional (default: 'metrics.json')
        Name of metrics file

    Returns
    -------
    metrics : ``dict``
        Dictionary of metrics
    """
    with open(os.path.join(exp_dir, filename), 'r') as f:
        return json.load(f)


def save_metrics(metrics, exp_dir, filename='metrics.json'):
    """
    Load metrics from the given exp directory..

    Parameters
    ----------
    metrics : ``dict``
        Metrics to save
    exp_dir : ``str``
        Folder to load metrics from
    filename : ``str``, optional (default: 'metrics.json')
        Name of metrics file
    """
    with open(os.path.join(exp_dir, filename), 'w') as f:
        json.dump(dict(metrics), f)


def parse_args(script, desc='', **kwargs):
    """
    Parse arguments for the given script using the argparse library.

    Parameters
    ----------
    script : ``str``
        Name of the script that is calling this function. This is used because
        different scripts/use cases (e.g. training, testing, visualization) may
        require different sets of arguments, yet they will all share a common
        set of arguments. I also normally have an option called 'interactive'
        which, if specified, does not take arguments from the command line, but
        rather takes arguments specified by **kwargs (e.g. if running in an
        iPython notebook).
    desc : ``str``, optional (default: "")
        Optional description of the script which is output when the `--help`
        flag is supplied via command line.
    **kwargs :
        If ``script == 'interactive'``, argparse will not query the command
        line for arguments but simply use those supplied by ``kwargs``.

    Returns
    -------
    args : ``argparse.Namespace``
        Namespace containing the parsed arguments

    Raises
    ------
    NotImplementedError :
        If ``script`` is unknown
    RuntimeError :
        If any ``**kwargs`` are specified but ``script != 'interactive'``
    """
    if script not in ['interactive', 'train']:
        raise NotImplementedError('script name = {}'.format(script))

    if kwargs and script != 'interactive':
        raise RuntimeError("Must run interactive to use kwargs")

    parser = ArgumentParser(
        description=desc,
        # Provide defaults in help message
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    # Args common to all scripts
    common_parser = parser.add_argument_group('common args')
    # Specify the data file to load.
    common_parser.add_argument('--data_file', default='data/data_3w_2b.csv',
                               help='Dataset to use')
    # Specify the experiment directory where all results/models from this run
    # are saved.
    common_parser.add_argument('--exp_dir', default='exp/debug/',
                               help='Folder to save experiment results')
    # Important: seed your experiments for reproducibility!
    common_parser.add_argument('--seed', type=int, default=42,
                               help='Random seed')
    # Most deep learning projects will use the GPU, but you want to be able to
    # eval off GPU if necessary. `--cuda` activates the GPU.
    common_parser.add_argument('--cuda', action='store_true',
                               help='Use cuda')
    # Often it's useful to have a `--debug` flag which will load a much simpler
    # dataset/model that is quicker to run, so you can diagnose problems.
    common_parser.add_argument('--debug', action='store_true')

    # Here we add script-specific arguments. It's a good idea to create an
    # argument group (`parser.add_argument_group`) to visually separate your
    # arguments when someone runs your script with the `--help` command.
    if script == 'train':
        train_parser = parser.add_argument_group('train args')

        # E.g. in the train script you may specify the model configuration; in
        # other scripts you can load the arguments saved during training; this
        # prevents you from having to remember the precise details of the model
        # when you load it later
        train_parser.add_argument('--init_m', type=float, default=1.0, help='Initial guess of m')
        train_parser.add_argument('--init_b', type=float, default=1.0, help='Initial guess of b')

        # Training details
        train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
        train_parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
        train_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

        # It is a REALLY great idea to make your script resumable; in case
        # you're running on a machine and something breaks, you won't lose all
        # your progress. It's a little annoying to figure out the logic
        # required to do so, but will save you headaches later on. (Machinery
        # for resuming is already written in this demo template)
        train_parser.add_argument('--resume', action='store_true', help='Resume from folder (if possible)')

        # It's helpful to log progress within epochs, esp. if epochs are long;
        # set a good default for the problem you're working on
        train_parser.add_argument('--log_interval', type=int, default=100,
                                  help='How often (in batches) to log progress in an epoch')

        train_parser.add_argument('--save_interval', type=int, default=100000000, help='How often (in epochs) to save models (besides last checkpoint/best model)')

        # Other helpers to speed up dataset loading if needed
        train_parser.add_argument('--n_workers', type=int, default=0, help='Number of dataloader workers')
        train_parser.add_argument('--pin_memory', action='store_true', help='Load data into CUDA-pinned memory')

    if kwargs:
        # XXX: This is a dumb way of converting kwargs into a single command
        # line string, which might break - be careful!
        args_str = [['--{}'.format(k), str(v)] if v is not True else ['--{}'.format(k)] for k, v in kwargs.items()]
        args_str = [item for sublist in args_str for item in sublist]
        args = parser.parse_args(args_str)
    else:
        args = parser.parse_args()


    # Often you need to perform some checks to make sure you've passed a
    # compatible set of arguments. Checking here can save you lots of pain
    # later in the experiment process, so it's a good idea to do so here!
    if script == 'train':
        if args.epochs > 1e100:
            parser.error('I refuse to run that long')

    return args

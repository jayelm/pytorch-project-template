"""
Data loading
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from torch.utils.data import Dataset


def load_raw_data(csv_file):
    """
    Load raw data from csv file.

    Parameters
    ----------
    data_file : ``str``
        filepath to data file (csv)

    Returns
    -------
    data : ``np.Array`` of shape ``(n_data, 2)``
        2D array where column 0 is input ``x`` and column 1 is binary output ``y``
    """
    csv_df = pd.read_csv(csv_file)
    return csv_df.values.astype(np.float32)


def train_val_test_split(data,
                         val_size=0.1,
                         test_size=0.1,
                         random_state=None,
                         **kwargs):
    """
    Split data into train, validation, and test splits

    Parameters
    ----------
    data : ``np.Array``
        Data of shape (n_data, 2), first column is ``x``, second column is ``y``
    val_size : ``float``, optional (default: 0.1)
        % to reserve for validation
    test_size : ``float``, optional (default: 0.1)
        % to reserve for test
    random_state : ``np.random.RandomState``, optional (default: None)
        If specified, random state for reproducibility
    kwargs :
        Additional keyword args to pass to ``Data`` wrapper
    """
    idx = np.arange(data.shape[0])
    idx_train, idx_valtest = train_test_split(idx,
                                              test_size=val_size + test_size,
                                              random_state=random_state,
                                              shuffle=True)
    idx_val, idx_test = train_test_split(idx_valtest,
                                         test_size=test_size /
                                         (val_size + test_size),
                                         random_state=random_state,
                                         shuffle=True)
    return {
        'train': Data(data[idx_train], **kwargs),
        'val': Data(data[idx_val], **kwargs),
        'test': Data(data[idx_test], **kwargs)
    }


class Data(Dataset):
    """
    Wrapper around dataset so that DataLoader can easily ingest it.
    """

    def __init__(self, data):
        """
        Initialize a Data object.

        Parameters
        ----------
        data : ``np.Array``
            Data of shape (n_data, 2), first column is ``x``, second column is ``y``
        """
        self.data = data

    def __getitem__(self, i):
        """
        Get the ith item of this object.

        Parameters
        -------
        i : ``int``
            Index to retrieve; must be within the bounds of the data

        Returns
        -------
        x : ``float``
            The input
        y : ``float``
            The binary output (1.0 or 0.0)
        """
        return tuple(self.data[i])

    def __len__(self):
        """
        Return the length of a Data object.

        Returns
        -------
        length : ``int``
            Number of datapints datapoints in this Data object
        """
        return self.data.shape[0]

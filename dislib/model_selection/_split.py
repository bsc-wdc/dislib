import numbers

from pycompss.api.task import task

from dislib import utils
from dislib.data import Dataset

import numpy as np


def infer_cv(cv=None):
    """Input checker utility for building a cross-validator
    Parameters
    ----------
    cv : int or splitter
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default KFold cross-validation splitter,
        - integer, to specify the number of folds,
        - custom CV splitter (must have the same interface as KFold).

    Returns
    -------
    checked_cv : a CV splitter instance.
        The return value is a CV splitter which generates the train/test
        splits via the ``split(dataset)`` method.
    """
    if cv is None:
        return KFold()
    if isinstance(cv, numbers.Integral):
        return KFold(cv)
    if not hasattr(cv, 'split') or not hasattr(cv, 'get_n_splits'):
        raise ValueError("Expected cv as an integer or splitter object."
                         "Got %s." % cv)
    return cv


class KFold:
    """K-fold splitter for cross-validation

    Returns k partitions of the dataset into train and validation datasets. The
    dataset is shuffled and split into k folds; each fold is used once as
    validation dataset while the k - 1 remaining folds form the training
    dataset.

    Each fold contains n//k or n//k + 1 samples, where n is the number of
    samples in the input dataset.

    Parameters
    ----------
    n_splits : int, optional (default=5)
        Number of folds. Must be at least 2.
    shuffle : boolean, optional (default=False)
        Shuffles and balances the data before splitting into batches.
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError('The number of folds must be of Integral type. '
                             '%s of type %s was passed.'
                             % (n_splits, type(n_splits)))
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits))

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False;"
                            " got {0}".format(shuffle))

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, dataset):
        """Generates K-fold splits.

        Parameters
        ----------
        dataset : Dataset
            Training data.

        Yields
        ------
        train_ds : Dataset
            The training dataset for that split.
        test_ds : Dataset
            The testing dataset for that split.
        """
        n_subsets_in = len(dataset)
        k = self.n_splits
        if self.shuffle:
            # Take n_subsets_out multiple of k and >= n_subsets_in
            n_subsets_out = n_subsets_in + (k - n_subsets_in % k)
            dataset = utils.shuffle(dataset, n_subsets_out, self.random_state)
        dataset, subsets_per_fold = _split_on_kfolds(dataset, k)

        start = 0
        for n_subsets in subsets_per_fold:
            end = start + n_subsets
            train_ds = dataset[0:start].concatenate(dataset[end:])
            test_ds = dataset[start:end]
            start = end
            yield train_ds, test_ds

    def get_n_splits(self):
        """Get the number of CV splits that this splitter does.

        Returns
        ------
        n_splits : Dataset
            The number of splits performed by this CV splitter.
        """
        return self.n_splits


def _split_on_kfolds(dataset, k):
    # Splits a dataset into k balanced folds, trying to avoid splitting subsets
    sizes = dataset.subsets_sizes()
    n = sum(sizes)

    base_size = n // k
    extended_folds = n % k  # Number of folds with n//k + 1 samples
    base_folds = k - extended_folds  # Number of folds with n//k samples

    subsets_per_fold = np.zeros(k, dtype=int)
    split_dataset = Dataset(dataset.n_features, dataset.sparse)
    fold_size = 0
    fold_idx = 0
    remaining = list(zip(dataset, sizes))
    remaining.reverse()
    while remaining:
        subset, size = remaining.pop()
        fold_size = fold_size + size
        if fold_size == base_size and base_folds > 0:
            # Complete the fold with the subset
            split_dataset.append(subset, size)
            subsets_per_fold[fold_idx] += 1

            base_folds -= 1
            fold_idx += 1
            fold_size = 0
        elif fold_size == base_size + 1 and extended_folds > 0:
            # Complete the fold with the subset
            split_dataset.append(subset, size)
            subsets_per_fold[fold_idx] += 1

            extended_folds -= 1
            fold_idx += 1
            fold_size = 0
        elif fold_size > base_size:
            # Split the subset to complete the fold
            if base_folds > 0:
                target_size = base_size
                base_folds -= 1
            else:  # n_extended_folds > 0
                target_size = base_size + 1
                extended_folds -= 1
            size_remainder = fold_size - target_size
            size_taken = size - size_remainder
            subset_0, subset_1 = _split_subset(subset, size_taken)
            split_dataset.append(subset_0, size_taken)
            subsets_per_fold[fold_idx] += 1
            fold_idx += 1
            fold_size = 0
            remaining.append((subset_1, size_remainder))
        else:
            # Add the subset and keep filling the fold
            split_dataset.append(subset, size)
            subsets_per_fold[fold_idx] += 1

    return split_dataset, subsets_per_fold


@task(returns=2)
def _split_subset(subset, split_size):
    return subset[:split_size], subset[split_size:]

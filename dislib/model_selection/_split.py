import numbers

import numpy as np
from pycompss.api.constraint import constraint
from pycompss.api.parameter import Type, COLLECTION_IN, Depth, COLLECTION_OUT
from pycompss.api.task import task

from dislib import utils
from dislib.data.array import Array


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

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
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

    def split(self, x, y=None):
        """Generates K-fold splits.

        Parameters
        ----------
        x : ds-array
            Samples array.
        y : ds-array, optional (default=None)
            Corresponding labels or values.

        Yields
        ------
        train_data : train_x, train_y
            The training ds-arrays for that split. If y is None, train_y is
            None.
        test_data : test_x, test_y
            The testing ds-arrays data for that split. If y is None, test_y is
            None.
        """
        k = self.n_splits
        if self.shuffle:
            shuffled = utils.shuffle(x, y, self.random_state)
            if y is None:
                x = shuffled
            else:
                x, y = shuffled
        n_total = x.shape[0]
        n_each_section, extras = divmod(n_total, k)
        section_sizes = np.empty((k,), dtype=int)
        section_sizes[:extras] = n_each_section + 1
        section_sizes[extras:] = n_each_section
        div_points = np.cumsum(section_sizes)
        yield get_kfold_partition(x, y, 0, div_points[0])
        for i in range(1, k):
            yield get_kfold_partition(x, y, div_points[i - 1], div_points[i])

    def get_n_splits(self):
        """Get the number of CV splits that this splitter does.

        Returns
        ------
        n_splits : int
            The number of splits performed by this CV splitter.
        """
        return self.n_splits


def get_kfold_partition(x, y, start, end):
    train_x = merge_slices(x[:start], x[end:])
    test_x = x[start:end]
    train_y = None
    test_y = None
    if y is not None:
        train_y = merge_slices(y[:start], y[end:])
        test_y = y[start:end]
    return (train_x, train_y), (test_x, test_y)


def merge_slices(s1, s2):
    """Merges horizontal slices s1 and s2 of an array. It works as in a
    concatenation, but the order of rows may change."""
    assert s1._shape[1] == s2._shape[1], """The arrays must have the same
     number of columns."""
    assert s1._sparse == s2._sparse, """A sparse and a dense array cannot
     be merged."""
    assert s1._reg_shape == s2._reg_shape, """The array regular blocks must
    have the same shape."""

    len_s1 = s1.shape[0]
    len_s2 = s2.shape[0]

    # If s1 or s2 is empty, quickly return the other slice.
    if len_s1 == 0:
        return s2
    if len_s2 == 0:
        return s1

    reg_shape = s1._reg_shape
    reg_rows = reg_shape[0]

    # Compute the start and end of regular row blocks for s1
    top_rows_s1 = s1._top_left_shape[0]
    reg_rows_start_s1 = top_rows_s1 if top_rows_s1 != reg_rows else 0
    reg_rows_end_s1 = len_s1 - (len_s1 - reg_rows_start_s1) % reg_rows

    # Compute the start and end of regular row blocks for s2
    top_rows_s2 = s2._top_left_shape[0]
    reg_rows_start_s2 = top_rows_s2 if top_rows_s2 != reg_rows else 0
    reg_rows_end_s2 = len_s2 - (len_s2 - reg_rows_start_s2) % reg_rows

    # Get arrays with the regular row blocks for s1 and s2
    reg_s1 = s1[reg_rows_start_s1:reg_rows_end_s1]
    reg_s2 = s2[reg_rows_start_s2:reg_rows_end_s2]

    # Add the regular row blocks to the list all_blocks
    all_blocks = []
    if reg_s1.shape[0]:
        all_blocks.extend(reg_s1._blocks)
    if reg_s2.shape[0]:
        all_blocks.extend(reg_s2._blocks)

    # If there are remaining rows on the top or bottom of s1 and s2, add them
    # to the list extras. These are row blocks with less than reg_rows.
    extras = []
    if reg_rows_start_s1 > 0:
        extras.append(s1[:reg_rows_start_s1])
    if reg_rows_start_s2 > 0:
        extras.append(s1[:reg_rows_start_s2])
    if reg_rows_end_s1 < len_s1:
        extras.append(s1[reg_rows_end_s1:])
    if reg_rows_end_s2 < len_s2:
        extras.append(s2[reg_rows_end_s2:])

    # Arrange the rows of the arrays in extras in groups of reg_rows rows,
    # slicing the arrays when necessary. The last group may have less than
    # reg_rows rows.
    groups = []
    current_capacity = 0
    for extra in extras:
        len_extra = extra.shape[0]
        if current_capacity == 0:
            current_capacity = reg_rows
            groups.append([])
        if extra.shape[0] <= current_capacity:
            current_capacity -= extra.shape[0]
            groups[-1].append(extra)
        else:
            groups[-1].append(extra[:current_capacity])
            groups.append([extra[current_capacity:]])
            current_capacity = current_capacity - len_extra + reg_rows

    # Merge the row blocks in each group, forming a single row block per group,
    # and add it to the list all blocks.
    for g in groups:
        blocks = []
        for a in g:
            for row_block in a._blocks:
                blocks.append(row_block)
        group_blocks = [object() for _ in range(s1._n_blocks[1])]
        _merge_rows_keeping_cols(blocks, group_blocks)
        all_blocks.append(group_blocks)

    # Now all_blocks contains all the rows of s1 and s2 in an appropiate
    # arrangement to create the merged array.
    return Array(blocks=all_blocks, top_left_shape=reg_shape,
                 reg_shape=reg_shape, shape=(len_s1 + len_s2, s1.shape[1]),
                 sparse=s1._sparse)


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_OUT, Depth: 1})
def _merge_rows_keeping_cols(blocks, out_blocks):
    """
    Merges the blocks vertically, into a single list of blocks (the number of
    cols per block is not modified).
    """
    left_cols = blocks[0][0].shape[1]
    reg_cols = blocks[0][1].shape[1] if len(blocks[0]) > 1 else None
    n_col_blocks = len(out_blocks)
    data = Array._merge_blocks(blocks)
    left = 0
    right = left_cols
    out_blocks[0] = data[:, left:right]
    for j in range(1, n_col_blocks):
        left = right
        right = right + reg_cols
        out_blocks[j] = data[:, left:right]

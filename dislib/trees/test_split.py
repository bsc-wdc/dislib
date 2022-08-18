from sys import float_info

import numpy as np


def criteria_proxy(l_weight, l_length, r_weight, r_length, not_repeated):
    """
    Maximizing the MSE or Gini gain is equivalent to minimizing
    this proxy function.
    """
    return -(l_weight / l_length + r_weight / r_length) * not_repeated


def test_split(sample, y_s, feature, n_classes):
    size = y_s.shape[0]
    if size == 0:
        return float_info.max, np.float64(np.inf)

    f = feature[sample]
    sort_indices = np.argsort(f)
    y_sorted = y_s[sort_indices]
    f_sorted = f[sort_indices]

    # Threshold value must not be that value of a sample
    not_repeated = np.empty(size, dtype=np.bool_)
    not_repeated[0: size - 1] = f_sorted[1:] != f_sorted[:-1]
    not_repeated[size - 1] = True

    if n_classes is not None:  # Classification
        l_freq = np.zeros((n_classes, size), dtype=np.int64)
        l_freq[y_sorted, np.arange(size)] = 1

        r_freq = np.zeros((n_classes, size), dtype=np.int64)
        r_freq[:, 1:] = l_freq[:, :0:-1]

        l_weight = np.sum(np.square(np.cumsum(l_freq, axis=-1)), axis=0)
        r_weight = np.sum(np.square(np.cumsum(r_freq, axis=-1)), axis=0)[::-1]

    else:  # Regression
        # Square of the sum of the y values of each branch
        r_weight = np.zeros(size)
        l_weight = np.square(np.cumsum(y_sorted, axis=-1))
        r_weight[:-1] = np.square(np.cumsum(y_sorted[::-1], axis=-1)[-2::-1])

    # Number of samples of each branch
    l_length = np.arange(1, size + 1, dtype=np.int32)
    r_length = np.arange(size - 1, -1, -1, dtype=np.int32)
    r_length[size - 1] = 1  # Avoid div by zero, the right score is 0

    scores = criteria_proxy(
        l_weight, l_length, r_weight, r_length, not_repeated
    )

    min_index = size - np.argmin(scores[::-1]) - 1
    if min_index + 1 == size:
        b_value = np.float64(np.inf)
    else:
        b_value = (f_sorted[min_index] + f_sorted[min_index + 1]) / 2
    return scores[min_index], b_value

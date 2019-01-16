from collections import defaultdict

import numpy as np
from pycompss.api.task import task
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components


class Region(object):

    def __init__(self, region_id, subset, n_samples, epsilon):
        self.id = region_id
        self.epsilon = epsilon
        self._neighbours = []
        self.subset = subset
        self.n_samples = n_samples
        self.labels = None
        self._neighbour_labels = []
        self._neighbour_ids = []

    def add_neighbour(self, region):
        self._neighbours.append(region)

    def partial_dbscan(self, min_samples, max_samples):
        subsets = [self.subset]
        n_samples = self.n_samples

        # get samples from all neighbouring regions
        for region in self._neighbours:
            subsets.append(region.subset)
            n_samples += region.n_samples

        if n_samples == 0:
            self.labels = np.empty(0)
            return

        # if max_samples is not defined, process all samples in a single task
        if max_samples is None:
            max_samples = n_samples

        # compute the neighbours of each sample using multiple tasks
        neigh_list = []
        cp_list = []

        for idx in range(0, n_samples, max_samples):
            end_idx = idx + max_samples
            neighs, cps = _compute_neighbours(self.epsilon, min_samples, idx,
                                              end_idx, *subsets)
            neigh_list.append(neighs)
            cp_list.append(cps)

        c_points = _lists_to_array(*cp_list)

        # compute the label of each sample based on their neighbours
        labels = _compute_labels(min_samples, n_samples, c_points, *neigh_list)
        self.labels = _slice_array(labels, 0, self.n_samples)

        # send labels to each neighbouring region
        start = self.n_samples
        finish = start

        for region in self._neighbours:
            finish += region.n_samples
            neigh_labels = _slice_array(labels, start, finish)
            region.add_labels(neigh_labels, self.id)
            start = finish

    def add_labels(self, labels, region_id):
        self._neighbour_labels.append(labels)
        self._neighbour_ids.append(region_id)

    def get_equivalences(self):
        return _compute_equivalences(self.id, self.labels, self._neighbour_ids,
                                     *self._neighbour_labels)

    def update_labels(self, components):
        self.labels = _update_labels(self.id, self.labels, components)


@task(returns=1)
def _update_labels(region_id, labels, components):
    new_labels = np.full(labels.shape[0], -1, dtype=int)

    for label, component in enumerate(components):
        for key in component:
            if key[:2] == region_id:
                indices = np.argwhere(labels == key[2])
                new_labels[indices] = label

    return new_labels


@task(returns=1)
def _compute_equivalences(region_id, labels, neigh_ids, *labels_list):
    equiv = defaultdict(set)

    for label_idx, label in enumerate(labels):
        if label < 0:
            continue

        key = region_id + (label,)

        if key not in equiv:
            equiv[key] = set()

        for neigh_id, neigh_labels in zip(neigh_ids, labels_list):
            neigh_label = neigh_labels[label_idx]

            if neigh_label >= 0:
                neigh_key = neigh_id + (neigh_label,)
                equiv[key].add(neigh_key)

    return equiv


@task(returns=1)
def _slice_array(arr, start, finish):
    return arr[start:finish]


@task(returns=2)
def _compute_neighbours(epsilon, min_samples, begin_idx, end_idx, *subsets):
    neighbour_list = []
    core_points = []
    samples = _concatenate_subsets(*subsets).samples

    for sample in samples[begin_idx:end_idx]:
        neighbours = np.linalg.norm(samples - sample, axis=1) < epsilon
        neigh_indices = np.where(neighbours)[0]
        neighbour_list.append(neigh_indices)
        core_points.append(neigh_indices.size >= min_samples)

    return neighbour_list, core_points


def _concatenate_subsets(*subsets):
    subset = subsets[0].copy()

    for set in subsets[1:]:
        subset.concatenate(set)

    return subset


@task(returns=1)
def _lists_to_array(*cp_list):
    return np.concatenate(cp_list)


@task(returns=1)
def _compute_labels(min_samples, n_samples, core_points, *neighbour_lists):
    adj_matrix = lil_matrix((n_samples, n_samples))
    row_idx = 0

    # Build adjacency matrix such that non-core points have a single
    # core point as neighbour. In this manner, non-core points will have
    # weak connections to all neighbours except one of the core points.
    for neighbour_list in neighbour_lists:
        for neighbours in neighbour_list:
            if core_points[row_idx]:
                adj_matrix.rows[row_idx] = neighbours
                adj_matrix.data[row_idx] = [1] * len(neighbours)
            elif len(neighbours) > 0:
                for neighbour in neighbours:
                    if core_points[neighbour]:
                        adj_matrix.rows[row_idx].append(neighbour)
                        adj_matrix.data[row_idx].append(1)
                        break

            row_idx += 1

    # ignores weak connections from core points to non-core points
    components, labels = connected_components(adj_matrix, connection="strong")

    for label in range(components):
        if labels[labels == label].size < min_samples:
            labels[labels == label] = -1

    return labels

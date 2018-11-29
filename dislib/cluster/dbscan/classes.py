from collections import defaultdict

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task

_CORE_POINT = -2
_NOISE = -1
_NO_CP = -3


class Region(object):
    def __init__(self, coord, epsilon, grid_shape, region_sizes):
        self.coord = coord
        self.epsilon = epsilon
        self.len_tot = 0
        self.offset = defaultdict()
        self.len = defaultdict()
        self._neigh_squares_query(region_sizes, grid_shape)
        self.subset = None
        self.cluster_labels = defaultdict(list)

    def init_data(self, data, grid_shape):
        prev = 0
        partitions = []

        for comb in self.neigh_sq_id:
            self.offset[comb] = prev
            part = data[np.ravel_multi_index(comb, grid_shape)]
            self.len[comb] = compss_wait_on(_count_lines(part))
            partitions.append(part)
            prev += self.len[comb]
            self.len_tot += self.len[comb]

        self.subset = _concatenate_data(*partitions)
        self._set_neigh_thres()

    def partial_scan(self, min_samples, max_samples):
        neigh_list = []

        if max_samples is None:
            max_samples = self.len_tot

        for idx in range(0, self.len_tot, max_samples):
            partial_list = _compute_neighbours(self.subset, self.epsilon, idx,
                                               idx + max_samples)
            neigh_list.append(partial_list)

        labels, self.core_points = _compute_labels(min_samples, *neigh_list)

        for neigh_id in self.neigh_sq_id:
            neigh_labels = _get_neigh_labels(labels,
                                             self.neigh_thres[neigh_id])
            self.cluster_labels[neigh_id] = neigh_labels

    def _set_neigh_thres(self):
        out = defaultdict(list)
        for comb in self.neigh_sq_id:
            out[comb] = [self.offset[comb], self.offset[comb] + self.len[comb]]
        self.neigh_thres = out

    def _neigh_squares_query(self, region_sizes, grid_shape):
        distances = np.ceil(self.epsilon / region_sizes)
        neigh_squares = []

        for ind in np.ndindex(grid_shape):
            d = np.abs(np.array(self.coord) - np.array(ind))

            if (d <= distances).all():
                neigh_squares.append(ind)

        self.neigh_sq_id = tuple(neigh_squares)


@task(returns=1)
def _compute_neighbours(subset, epsilon, begin_idx, end_idx):
    neighbour_list = []
    samples = subset.samples

    for sample in samples[begin_idx:end_idx]:
        neighbours = np.linalg.norm(samples - sample, axis=1) < epsilon
        neigh_indices = np.where(neighbours)[0]
        neighbour_list.append(neigh_indices)

    return neighbour_list


@task(returns=2)
def _compute_labels(min_samples, *neighbour_lists):
    final_list = neighbour_lists[0]

    for neighbour_list in neighbour_lists[1:]:
        final_list.extend(neighbour_list)

    clusters, core_points = _compute_clusters(final_list, min_samples)
    labels = np.full(len(final_list), _NOISE)

    for cluster_id, sample_indices in enumerate(clusters):
        labels[sample_indices] = cluster_id

    return labels, core_points


@task(returns=1)
def _get_neigh_labels(labels, indices):
    return labels[indices[0]:indices[1]]


def _compute_clusters(neigh_list, min_samples):
    visited = []
    clusters = []
    core_points = np.full(len(neigh_list), _NO_CP)

    for sample_idx, neighs in enumerate(neigh_list):
        if sample_idx in visited:
            continue

        if neighs.size >= min_samples:
            clusters.append([sample_idx])
            visited.append(sample_idx)
            core_points[sample_idx] = _CORE_POINT
            _visit_neighbours(neigh_list, neighs, visited, clusters,
                              core_points, min_samples)

    return clusters, core_points


def _visit_neighbours(neigh_list, neighbours, visited, clusters, core_points,
                      min_samples):
    for neigh_idx in neighbours:
        if neigh_idx in visited:
            continue

        visited.append(neigh_idx)
        clusters[-1].append(neigh_idx)

        if neigh_list[neigh_idx].size >= min_samples:
            core_points[neigh_idx] = _CORE_POINT
            new_neighbours = neigh_list[neigh_idx]

            _visit_neighbours(neigh_list, new_neighbours, visited, clusters,
                              core_points, min_samples)


def _get_connected_components(transitions):
    visited = []
    connected = []
    for node, neighbours in transitions.items():
        if node in visited:
            continue

        connected.append([node])

        _visit_neighbours(transitions, neighbours, visited, connected)
    return connected


@task(returns=1)
def _merge_core_points(chunks, comb, *cp_list):
    tmp = [max(i) for i in list(zip(*cp_list))]
    return tmp[chunks[comb][0]: chunks[comb][1]]


@task(returns=1)
def _concatenate_data(*subsets):
    set0 = subsets[0]

    for set in subsets[1:]:
        set0.concatenate(set)

    return set0


@task(returns=int)
def _count_lines(subset):
    return subset.samples.shape[0]

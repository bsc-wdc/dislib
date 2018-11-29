from collections import defaultdict

import numpy as np
from pycompss.api.api import compss_wait_on

from dislib.cluster.dbscan.tasks import count_lines, concatenate_data, \
    _compute_neighbours, _compute_labels, _get_neigh_labels


class Square(object):
    def __init__(self, coord, epsilon, grid_shape, region_sizes):
        self.coord = coord
        self.epsilon = epsilon
        self.len_tot = 0
        self.offset = defaultdict()
        self.len = defaultdict()
        self._neigh_squares_query(region_sizes, grid_shape)
        self.subset = None
        self.cluster_labels = defaultdict(list)

    def _neigh_squares_query(self, region_sizes, grid_shape):
        distances = np.ceil(self.epsilon / region_sizes)
        neigh_squares = []

        for ind in np.ndindex(grid_shape):
            d = np.abs(np.array(self.coord) - np.array(ind))

            if (d <= distances).all():
                neigh_squares.append(ind)

        self.neigh_sq_id = tuple(neigh_squares)

    def init_data(self, data, grid_shape):
        prev = 0
        partitions = []

        for comb in self.neigh_sq_id:
            self.offset[comb] = prev
            part = data[np.ravel_multi_index(comb, grid_shape)]
            self.len[comb] = compss_wait_on(count_lines(part))
            partitions.append(part)
            prev += self.len[comb]
            self.len_tot += self.len[comb]

        self.subset = concatenate_data(*partitions)
        self._set_neigh_thres()

    def _set_neigh_thres(self):
        out = defaultdict(list)
        for comb in self.neigh_sq_id:
            out[comb] = [self.offset[comb], self.offset[comb] + self.len[comb]]
        self.neigh_thres = out

    def _partial_scan(self, min_samples, max_samples):
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

from collections import defaultdict

import numpy as np
from pycompss.api.api import compss_wait_on

from dislib.cluster.dbscan.tasks import count_lines, concatenate_data, \
    orq_scan_merge, merge_relations, merge_cluster_labels, merge_core_points, \
    sync_task, update_task


class Square(object):
    def __init__(self, coord, epsilon, grid_shape, region_sizes):
        self.coord = coord
        self.epsilon = epsilon
        self.len_tot = 0
        self.offset = defaultdict()
        self.len = defaultdict()
        self._neigh_squares_query(region_sizes, grid_shape)

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

        self.points = concatenate_data(*partitions)
        self._set_neigh_thres()

    def _set_neigh_thres(self):
        out = defaultdict(list)
        for comb in self.neigh_sq_id:
            out[comb] = [self.offset[comb], self.offset[comb] + self.len[comb]]
        self.neigh_thres = out

    def partial_scan(self, min_points, TH_1):
        label_list, cp_list = orq_scan_merge(self.points, self.epsilon,
                                             min_points, TH_1, 1, 0, [], [],
                                             self.len_tot)



        self.cluster_labels = defaultdict(list)

        for comb in self.neigh_sq_id:
            self.cluster_labels[comb] = merge_cluster_labels(self.neigh_thres[
                                                                 comb],
                                                             *label_list)

        self.core_points = merge_core_points(self.neigh_thres, self.coord,
                                             *cp_list)

    def sync_labels(self, *labels_versions):
        return sync_task(self.coord, self.cluster_labels[self.coord],
                         self.core_points, self.neigh_sq_id, *labels_versions)

    def update_labels(self, updated_relations):
        self.cluster_labels[self.coord] = compss_wait_on(update_task(
            self.cluster_labels[self.coord], self.coord, updated_relations))

    def get_labels(self):
        return self.cluster_labels[self.coord]


class Data(object):
    def __init__(self):
        self.value = []


class DisjointSet:
    _disjoint_set = list()

    #    def __init__(self, init_arr):
    #        self._disjoint_set = []
    #        if init_arr:
    #            for item in list(set(init_arr)):
    #                self._disjoint_set.append([item])

    # Alternative __init__:
    def __init__(self, init_arr):
        self._disjoint_set = []
        if init_arr:
            for item in list(init_arr):
                self._disjoint_set.append([item])

    def _find_index(self, elem):
        for item in self._disjoint_set:
            if elem in item:
                return self._disjoint_set.index(item)
        return None

    def find(self, elem):
        for item in self._disjoint_set:
            if elem in item:
                return self._disjoint_set[self._disjoint_set.index(item)]
        return None

    def union(self, elem1, elem2):
        index_elem1 = self._find_index(elem1)
        index_elem2 = self._find_index(elem2)

        if index_elem1 != index_elem2 and \
           index_elem1 is not None and \
           index_elem2 is not None:
            self._disjoint_set[index_elem2] = self._disjoint_set[index_elem2] \
                                              + self._disjoint_set[index_elem1]

            del self._disjoint_set[index_elem1]

        return self._disjoint_set

    def get(self):
        return self._disjoint_set

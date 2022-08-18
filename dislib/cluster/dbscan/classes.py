import bisect
from collections import defaultdict
from itertools import chain

import numpy as np
from pycompss.api.constraint import constraint
from pycompss.api.task import task
from scipy.sparse import lil_matrix, vstack as vstack_sparse
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


class Region(object):

    def __init__(self, region_id, samples, n_samples, epsilon, sparse):
        self.id = region_id
        self.epsilon = epsilon
        self.samples = samples
        self.n_samples = n_samples
        self.labels_region = None
        self.labels = None
        self.cp_labels = None
        self._neigh_regions = []
        self._neigh_regions_ids = []
        self._in_cp_neighs = []
        self._out_cp_neighs = []
        self._non_cp_neighs = []
        self._sparse = sparse

    def add_neighbour(self, region):
        self._neigh_regions.append(region)

    def partial_dbscan(self, min_samples, max_samples):
        if self.n_samples == 0:
            self.cp_labels = np.empty(0, dtype=int)
            return

        neigh_samples = []
        total_n_samples = self.n_samples

        # get samples from all neighbouring regions
        for region in self._neigh_regions:
            neigh_samples.append(region.samples)
            total_n_samples += region.n_samples

        # if max_samples is not defined, process all samples in a single task
        if max_samples is None:
            max_samples = self.n_samples

        # compute the neighbours of each sample using multiple tasks
        cp_list = []

        for idx in range(0, self.n_samples, max_samples):
            end_idx = idx + max_samples
            result = _compute_neighbours(self.epsilon, min_samples, idx,
                                         end_idx, self.samples, self._sparse,
                                         *neigh_samples)
            cps, in_cp_neighs, out_cp_neighs, non_cp_neighs = result
            cp_list.append(cps)
            self._in_cp_neighs.append(in_cp_neighs)
            self._out_cp_neighs.append(out_cp_neighs)
            self._non_cp_neighs.append(non_cp_neighs)

        cp_mask = _lists_to_array(*cp_list)

        # perform a local DBSCAN clustering on the core points
        self.cp_labels = _compute_cp_labels(cp_mask, *self._in_cp_neighs)

    def get_equivalences(self):
        if self.n_samples == 0:
            self.labels_region = np.empty(0)
            self.labels = np.empty(0)
            return {}
        # get samples from all neighbouring regions
        regions_ids = [self.id]
        cp_labels_list = [self.cp_labels]
        for region in self._neigh_regions:
            regions_ids.append(region.id)
            cp_labels_list.append(region.cp_labels)
        result = _compute_equivalences(self.n_samples, regions_ids,
                                       *chain(cp_labels_list,
                                              self._out_cp_neighs,
                                              self._non_cp_neighs))
        self.labels_region, self.labels, equiv = result

        return equiv

    def update_labels(self, components):
        self.labels = _update_labels(self.labels_region, self.labels,
                                     components)


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _update_labels(labels_region, labels, components):
    components_map = {}
    for label, component in enumerate(components):
        for key in component:
            components_map[key] = label

    for i, (r, lbl) in enumerate(zip(labels_region, labels)):
        labels[i] = components_map.get(tuple(r) + (lbl,), lbl)

    return labels


@constraint(computing_units="${ComputingUnits}")
@task(returns=3)
def _compute_equivalences(n_samples, region_ids, *starred_args):
    n_regions = len(region_ids)
    cp_labels_list = starred_args[:n_regions]
    n_chunks = len(starred_args[n_regions:]) // 2
    out_cp_neighs = iter(chain(*starred_args[n_regions:n_regions + n_chunks]))
    non_cp_neighs = iter(chain(*starred_args[n_regions + n_chunks:]))

    region_id = region_ids[0]
    region_cp_labels = cp_labels_list[0]

    labels = region_cp_labels.copy()
    label_regions = np.empty((n_samples, len(region_id)), dtype=int)
    label_regions[:] = region_id

    equiv = defaultdict(set)
    for idx in range(n_samples):
        if region_cp_labels[idx] != -1:  # if core point
            # Add equivalences to neighbouring clusters of other regions
            out_neighbours = next(out_cp_neighs)
            key = region_id + (region_cp_labels[idx],)
            if key not in equiv:
                equiv[key] = set()
            for n in out_neighbours:
                if n[0] > 0 and cp_labels_list[n[0]][n[1]] != -1:
                    n_region_id = region_ids[n[0]]
                    n_label = cp_labels_list[n[0]][n[1]]
                    equiv[key].add(n_region_id + (n_label,))
        else:
            # Assign the label of the closest core point neighbour (if exists)
            neighbours = next(non_cp_neighs)
            for n in neighbours:
                if cp_labels_list[n[0]][n[1]] != -1:  # if core point
                    n_region_id = region_ids[n[0]]
                    n_label = cp_labels_list[n[0]][n[1]]
                    label_regions[idx] = n_region_id
                    labels[idx] = n_label
                    break
    return label_regions, labels, equiv


@constraint(computing_units="${ComputingUnits}")
@task(returns=4)
def _compute_neighbours(epsilon, min_samples, begin_idx, end_idx, samples,
                        sparse, *neigh_samples):
    all_len = [samples.shape[0]] + [s.shape[0] for s in neigh_samples]
    cum_len = np.cumsum(all_len)
    all_samples = _concatenate_samples(sparse, samples, *neigh_samples)
    nn = NearestNeighbors(radius=epsilon)
    nn.fit(all_samples)
    dists, neighs = nn.radius_neighbors(samples[begin_idx:end_idx],
                                        return_distance=True)
    core_points = [len(neighbors) >= min_samples for neighbors in neighs]

    inner_core_neighbors = []
    outer_core_neighbors = []
    noncore_neighbors = []
    for i, (distances, neighbors) in enumerate(zip(dists, neighs)):
        idx = begin_idx + i
        if core_points[i]:
            neighbors_in = []
            neighbors_out = []
            for n in neighbors:
                if n != idx:
                    reg = bisect.bisect(cum_len, n)
                    if reg == 0:
                        if idx < n:
                            neighbors_in.append(n)
                    else:
                        reg_idx = n - cum_len[reg - 1]
                        if idx <= reg_idx:
                            neighbors_out.append((reg, reg_idx))
            inner_core_neighbors.append(np.array(neighbors_in, dtype=int))
            outer_core_neighbors.append(neighbors_out)
        else:
            neighbors = neighbors[np.argsort(distances)]
            neighbors_tups = []
            for n in neighbors:
                if n != idx:
                    reg = bisect.bisect(cum_len, n)
                    reg_idx = n if reg == 0 else n - cum_len[reg - 1]
                    neighbors_tups.append((reg, reg_idx))
            noncore_neighbors.append(neighbors_tups)

    return (core_points, inner_core_neighbors, outer_core_neighbors,
            noncore_neighbors)


def _concatenate_samples(sparse, *samples):
    if not sparse:
        return np.vstack(samples)
    else:
        return vstack_sparse(samples)


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _lists_to_array(*cp_list):
    return np.concatenate(cp_list)


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _compute_cp_labels(core_points, *in_cp_neighs):
    core_ids = np.cumsum(core_points) - 1
    n_core_pts = np.count_nonzero(core_points)
    adj_matrix = lil_matrix((n_core_pts, n_core_pts))

    # Build adjacency matrix of core points
    in_cp_neighs_iter = chain(*in_cp_neighs)
    core_idx = 0
    for idx, neighbours in zip(core_points.nonzero()[0], in_cp_neighs_iter):
        neighbours = core_ids[neighbours[core_points[neighbours]]]
        adj_matrix.rows[core_idx] = neighbours.tolist()
        adj_matrix.data[core_idx] = [1] * len(neighbours)
        core_idx += 1

    n_clusters, core_labels = connected_components(adj_matrix, directed=False)
    labels = np.full(core_points.shape, -1)
    labels[core_points] = core_labels
    return labels

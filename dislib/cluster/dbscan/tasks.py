import numpy as np
from pycompss.api.task import task

from dislib.cluster.dbscan.constants import CORE_POINT, NO_CP, NOISE


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
    labels = np.full(len(final_list), NOISE)

    for cluster_id, sample_indices in enumerate(clusters):
        labels[sample_indices] = cluster_id

    return labels, core_points


@task(returns=1)
def _get_neigh_labels(labels, indices):
    return labels[indices[0]:indices[1]]


def _compute_clusters(neigh_list, min_samples):
    visited = []
    clusters = []
    core_points = np.full(len(neigh_list), NO_CP)

    for sample_idx, neighs in enumerate(neigh_list):
        if sample_idx in visited:
            continue

        if neighs.size >= min_samples:
            clusters.append([sample_idx])
            visited.append(sample_idx)
            core_points[sample_idx] = CORE_POINT
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
            core_points[neigh_idx] = CORE_POINT
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
def merge_core_points(chunks, comb, *cp_list):
    tmp = [max(i) for i in list(zip(*cp_list))]
    return tmp[chunks[comb][0]: chunks[comb][1]]


@task(returns=1)
def concatenate_data(*subsets):
    set0 = subsets[0]

    for set in subsets[1:]:
        set0.concatenate(set)

    return set0


@task(returns=int)
def count_lines(subset):
    return subset.samples.shape[0]

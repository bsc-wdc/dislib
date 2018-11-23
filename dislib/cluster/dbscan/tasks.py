from collections import defaultdict

import numpy as np
from pycompss.api.task import task

from dislib.cluster.dbscan.constants import CORE_POINT, NOT_PROCESSED, NO_CP, \
    NOISE


@task(returns=list)
def merge_task_sync(adj_mat, *args):
    adj_mat_copy = [[] for _ in range(max(adj_mat[0], 1))]
    for args_i in args:
        for num, list_elem in enumerate(args_i):
            for elem in list_elem:
                if elem not in adj_mat_copy[num]:
                    adj_mat_copy[num].append(elem)
    return adj_mat_copy


@task(returns=1)
def sync_task(coord, cluster_labels, core_points, neigh_sq_id,
              *labels_versions):
    out = defaultdict(set)
    for num_label, label in enumerate(cluster_labels):
        if core_points[num_label] == CORE_POINT:
            # Current cluster label unique identifier
            point_id = (coord, cluster_labels[num_label])
            # Labels for the same point obtained by different workers
            point_versions = [vec[num_label] for vec in labels_versions]
            for num_dif, p_ver in enumerate(point_versions):
                out[point_id].add((neigh_sq_id[num_dif], p_ver))

    return out


@task(returns=1)
def update_task(cluster_labels, coord, updated_relations):
    direct_link = defaultdict()
    for num, label in enumerate(cluster_labels):
        if label in direct_link:
            cluster_labels[num] = direct_link[label]
        else:
            id_tuple = (coord, label)

            for num_list, _list in enumerate(updated_relations):
                if id_tuple in _list:
                    direct_link[label] = num_list
                    cluster_labels[num] = direct_link[label]
                    break

    return cluster_labels


def orq_scan_merge(data, epsilon, min_points, TH_1, quocient,
                   res, label_list, cp_list, len_total):
    if (len_total / quocient) > TH_1:
        label_list, cp_list = orq_scan_merge(data, epsilon, min_points, TH_1,
                                             quocient * 2, res * 2 + 0,
                                             label_list, cp_list, len_total)

        label_list, cp_list = orq_scan_merge(data, epsilon, min_points, TH_1,
                                             quocient * 2, res * 2 + 1,
                                             label_list, cp_list, len_total)
    else:
        labels, core_points = partial_dbscan(data, epsilon, min_points,
                                             quocient, res, len_total)

        label_list.append(labels)
        cp_list.append(core_points)

    return label_list, cp_list


@task(returns=2)
def partial_dbscan(subset, epsilon, min_points, quocient, res, len_tot):
    samples = subset.samples
    indices = [i for i in range(len_tot) if ((i % quocient) == res)]
    cluster_count = 0
    cluster_labels = np.array([NOT_PROCESSED] * len_tot)
    core_points = np.array([NO_CP] * len_tot)

    for i in indices:
        neigh_points = np.linalg.norm(samples - samples[i], axis=1) < epsilon
        neigh_sum = np.sum(neigh_points)

        if neigh_sum >= min_points:
            core_points[i] = CORE_POINT
            cluster_labels[i] = cluster_count
            neigh_idx = np.where(neigh_points)[0]

            for j in neigh_idx:
                neigh_label = cluster_labels[j]
                cluster_labels[j] = cluster_count

                if core_points[j] == CORE_POINT:
                    cluster_labels[
                        cluster_labels == neigh_label] = cluster_count

            cluster_count += 1
        elif cluster_labels[i] == NOT_PROCESSED:
            cluster_labels[i] = NOISE

    return cluster_labels, core_points


@task(returns=1)
def merge_cluster_labels(chunks, *labels_list):
    new_labels, transitions = _compute_transitions(labels_list)
    connected = _get_connected_components(transitions)

    for component in connected:
        min_ = min(component)

        for label in component:
            new_labels[new_labels == label] = min_

    return new_labels[chunks[0]: chunks[1]]


def _get_connected_components(transitions):
    visited = []
    connected = []
    for node, neighbours in transitions.items():
        if node in visited:
            continue

        connected.append([node])

        _visit_neighbours(transitions, neighbours, visited, connected)
    return connected


def _compute_transitions(labels_list):
    labels = np.array(labels_list)
    new_labels = np.full(labels[0].shape[0], -1)
    transitions = defaultdict(set)

    for i in range(len(new_labels)):
        final_indices = np.empty(0, dtype=int)

        for label_vec in labels[labels[:, i] >= 0]:
            indices = np.argwhere(label_vec == label_vec[i])[:, 0]
            final_indices = np.concatenate((final_indices, indices))

        final_indices = np.unique(final_indices)
        new_label = min(i, max(new_labels[i], i))

        trans = new_labels[final_indices]
        trans = np.unique(trans[trans >= 0])

        new_labels[final_indices] = new_label

        for label in trans:
            transitions[new_label].add(label)
            transitions[label].add(new_label)

    return new_labels, transitions


def _visit_neighbours(transitions, neighbours, visited, connected):
    for neighbour in neighbours:
        if neighbour in visited:
            continue

        visited.append(neighbour)
        connected[-1].append(neighbour)

        if neighbour in transitions:
            new_neighbours = transitions[neighbour]

            _visit_neighbours(transitions, new_neighbours, visited, connected)


@task(returns=1)
def merge_relations(*args):
    from dislib.cluster.dbscan.classes import DisjointSet
    out = defaultdict(set)
    for dic in args:
        for key in dic:
            out[key] |= dic[key]
    mf_set = DisjointSet(out.keys())
    for key in out:
        tmp = list(out[key])
        for i in range(len(tmp) - 1):
            # for j in range(i, len(tmp)):
            #     mf_set.union(tmp[i], tmp[j])
            mf_set.union(tmp[i], tmp[i + 1])
    out = mf_set.get()

    return out


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

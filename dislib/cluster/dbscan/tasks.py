from collections import defaultdict

import numpy as np
from pycompss.api.task import task

from dislib.cluster.dbscan.constants import *


def orquestrate_sync_clusters(data, adj_mat, epsilon, coord, neigh_sq_loc,
                              len_neighs, quocient, res, fut_list, TH_2,
                              count_tasks, *args):
    if (len_neighs / quocient) > TH_2:
        [fut_list,
         count_tasks] = orquestrate_sync_clusters(data, adj_mat, epsilon, coord,
                                                  neigh_sq_loc, len_neighs,
                                                  quocient * 2, res * 2 + 0,
                                                  fut_list,
                                                  TH_2, count_tasks, *args)
        [fut_list,
         count_tasks] = orquestrate_sync_clusters(data, adj_mat, epsilon, coord,
                                                  neigh_sq_loc, len_neighs,
                                                  quocient * 2, res * 2 + 1,
                                                  fut_list,
                                                  TH_2, count_tasks, *args)
    else:
        count_tasks += 1
        fut_list.append(orquestrate_sync_clusters(data, adj_mat, epsilon, coord,
                                                  neigh_sq_loc, quocient, res,
                                                  len_neighs,
                                                  *args))
    return fut_list, count_tasks


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
                   res, fut_list, len_total):
    if (len_total / quocient) > TH_1:
        [fut_list[0],
         fut_list[1],
         fut_list[2]] = orq_scan_merge(data, epsilon, min_points, TH_1,
                                       quocient * 2, res * 2 + 0,
                                       fut_list, len_total)

        [fut_list[0],
         fut_list[1],
         fut_list[2]] = orq_scan_merge(data, epsilon, min_points, TH_1,
                                       quocient * 2, res * 2 + 1,
                                       fut_list, len_total)
    else:
        obj = [[], [], []]
        obj[0], obj[1], obj[2] = partial_dbscan(data, epsilon, min_points,
                                                quocient, res, len_total)

        for num, _list in enumerate(fut_list):
            _list.append(obj[num])

    return fut_list[0], fut_list[1], fut_list[2]


@task(returns=3)
def partial_dbscan(data, epsilon, min_points, quocient, res, len_tot):
    points = data.vectors
    indices = [i for i in range(len_tot) if ((i % quocient) == res)]
    cluster_count = 0
    cluster_labels = np.array([NOT_PROCESSED] * len_tot)
    core_points = np.array([NO_CP] * len_tot)
    relations = defaultdict(set)

    for i in indices:
        neigh_points = np.linalg.norm(points - points[i], axis=1) < epsilon
        neigh_sum = np.sum(neigh_points)

        if neigh_sum >= min_points:
            core_points[i] = CORE_POINT
            cluster_labels[i] = cluster_count
            neigh_idx = np.where(neigh_points)

            for j in neigh_idx[0]:
                neigh_label = cluster_labels[j]
                cluster_labels[j] = cluster_count

                if core_points[j] == CORE_POINT:
                    cluster_labels[
                        cluster_labels == neigh_label] = cluster_count

            cluster_count += 1
        else:
            cluster_labels[i] = NOISE

    return cluster_labels, relations, core_points


@task(returns=1)
def merge_cluster_labels(chunks, *args):
    new_labels, transitions = _compute_transitions(args)

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


def _compute_transitions(args):
    labels = np.array(args)
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
def merge_core_points(chunks, comb, *args):
    tmp = [max(i) for i in list(zip(*args))]
    return tmp[chunks[comb][0]: chunks[comb][1]]


@task(returns=1)
def concatenate_data(*parts):
    ds = parts[0]

    for part in parts[1:]:
        ds.concatenate(part)

    return ds


@task(returns=int)
def count_lines(part):
    return part.vectors.shape[0]

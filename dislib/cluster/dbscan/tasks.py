import os
from collections import defaultdict

import numpy as np
from pandas import read_csv
from pycompss.api.parameter import FILE_IN
from pycompss.api.parameter import FILE_OUT
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


@task(file_path=FILE_OUT)
def update_task(cluster_labels, coord, points, thres, updated_relations,
                file_path):
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
    # Write output to files.
    f_out = open(file_path, "w")
    for num, val in enumerate(cluster_labels):
        f_out.write(str(points[thres + num]) + " "
                    + str(cluster_labels[num]) + "\n")
    f_out.close()


def orq_scan_merge(data, epsilon, min_points, TH_1, count_tasks, quocient,
                   res, fut_list, len_total):
    if (len_total / quocient) > TH_1:
        [fut_list[0],
         fut_list[1],
         fut_list[2],
         count_tasks] = orq_scan_merge(data, epsilon, min_points, TH_1,
                                       count_tasks, quocient * 2, res * 2 + 0,
                                       fut_list, len_total)
        [fut_list[0],
         fut_list[1],
         fut_list[2],
         count_tasks] = orq_scan_merge(data, epsilon, min_points, TH_1,
                                       count_tasks, quocient * 2, res * 2 + 1,
                                       fut_list, len_total)
    else:
        obj = [[], [], []]
        count_tasks += 1
        obj[0], obj[1], obj[2] = partial_dbscan(data, epsilon, min_points,
                                                quocient, res, len_total)
        for num, _list in enumerate(fut_list):
            _list.append(obj[num])
    return fut_list[0], fut_list[1], fut_list[2], count_tasks


@task(returns=3)
def partial_dbscan(data, epsilon, min_points, quocient, res, len_tot):
    indices = [i for i in range(len_tot) if ((i % quocient) == res)]
    cluster_labels = [NOT_PROCESSED for i in range(len_tot)]
    core_points = [NO_CP for i in range(len_tot)]
    cluster_count = 0
    relations = defaultdict(set)
    for i in indices:
        neigh_points = np.linalg.norm(data - data[i], axis=1) < epsilon
        neigh_sum = np.sum(neigh_points)
        if neigh_sum >= min_points:
            core_points[i] = CORE_POINT
            cluster_labels[i] = cluster_count
            neigh_idx = np.where(neigh_points)
            for j in neigh_idx[0]:
                if core_points[j] == CORE_POINT:
                    relations[cluster_count].add(cluster_labels[j])
                cluster_labels[j] = cluster_count
            cluster_count += 1
        else:
            cluster_labels[i] = NOISE
    return cluster_labels, relations, core_points


@task(returns=1)
def merge_cluster_labels(relations, comb, chunks, *args):
    tmp = [max(i) for i in list(zip(*args))]
    for i, label in enumerate(tmp):
        for num, _list in enumerate(relations):
            if label in _list:
                tmp[i] = num
    # We pre-chunk the cluster labels, so the labels of each region are
    # accesssible separately.
    return tmp[chunks[0]: chunks[1]]


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
            mf_set.union(tmp[i], tmp[i + 1])
    out = mf_set.get()
    return out


@task(returns=1)
def merge_core_points(chunks, comb, *args):
    tmp = [max(i) for i in list(zip(*args))]
    return tmp[chunks[comb][0]: chunks[comb][1]]


def count_lines(tupla, file_id, is_mn):
    if is_mn:
        path = "/gpfs/projects/bsc19/COMPSs_DATASETS/dbscan/" + str(file_id)
    else:
        path = "~/DBSCAN/data/" + str(file_id)
    path = os.path.expanduser(path)
    tmp_string = path + "/" + str(tupla[0])
    for num, j in enumerate(tupla):
        if num > 0:
            tmp_string += "_" + str(j)
    tmp_string += ".txt"
    with open(tmp_string) as infile:
        for i, line in enumerate(infile):
            pass
        return i + 1


def orquestrate_init_data(tupla, file_id, len_data, quocient,
                          res, fut_list, TH_1, is_mn):
    if (len_data / quocient) > TH_1:
        fut_list = orquestrate_init_data(tupla, file_id, len_data,
                                              quocient * 2, res * 2 + 0,
                                              fut_list,
                                              TH_1, is_mn)
        fut_list = orquestrate_init_data(tupla, file_id, len_data,
                                              quocient * 2, res * 2 + 1,
                                              fut_list,
                                              TH_1, is_mn)
    else:
        if is_mn:
            path = "/gpfs/projects/bsc19/COMPSs_DATASETS/dbscan/" + str(file_id)
        else:
            path = "~/DBSCAN/data/" + str(file_id)
        path = os.path.expanduser(path)
        tmp_string = path + "/" + str(tupla[0])
        for num, j in enumerate(tupla):
            if num > 0:
                tmp_string += "_" + str(j)
        tmp_string += ".txt"
        tmp_f = init_data(tmp_string, quocient, res, is_mn)
        fut_list.append(tmp_f)
    return fut_list


# @task(isDistributed=True, file_path=FILE_IN, returns=1)
@task(file_path=FILE_IN, returns=1)
def init_data(file_path, quocient, res, is_mn):
    df = read_csv(file_path, sep=' ', skiprows=lambda x: (x % quocient)
                                                         != res, header=None)
    data = df.values.astype(float)
    return data


@task(returns=1)
def merge_task_init_data(*args):
    tmp_data = np.vstack([i for i in args])
    return tmp_data

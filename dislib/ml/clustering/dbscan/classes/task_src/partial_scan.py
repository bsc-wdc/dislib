from collections import defaultdict
import numpy as np
from pycompss.api.task import task
import constants
from DS import DisjointSet


def orq_scan_merge(data, epsilon, min_points, TH_1, count_tasks, quocient,
                   res, fut_list, len_total):
    if (len_total/quocient) > TH_1:
        [fut_list[0],
         fut_list[1],
         fut_list[2],
         count_tasks] = orq_scan_merge(data, epsilon, min_points, TH_1,
                                       count_tasks, quocient*2, res*2 + 0,
                                       fut_list, len_total)
        [fut_list[0],
         fut_list[1],
         fut_list[2],
         count_tasks] = orq_scan_merge(data, epsilon, min_points, TH_1,
                                       count_tasks, quocient*2, res*2 + 1,
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
    cluster_labels = [constants.NOT_PROCESSED for i in range(len_tot)]
    core_points = [constants.NO_CP for i in range(len_tot)]
    cluster_count = 0
    relations = defaultdict(set)
    for i in indices:
        neigh_points = np.linalg.norm(data - data[i], axis=1) < epsilon
        neigh_sum = np.sum(neigh_points)
        if neigh_sum >= min_points:
            core_points[i] = constants.CORE_POINT
            cluster_labels[i] = cluster_count
            neigh_idx = np.where(neigh_points)
            for j in neigh_idx[0]:
                if core_points[j] == constants.CORE_POINT:
                    relations[cluster_count].add(cluster_labels[j])
                cluster_labels[j] = cluster_count
            cluster_count += 1
        else:
            cluster_labels[i] = constants.NOISE
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
    out = defaultdict(set)
    for dic in args:
        for key in dic:
            out[key] |= dic[key]
    mf_set = DisjointSet(out.keys())
    for key in out:
        tmp = list(out[key])
        for i in range(len(tmp)-1):
            mf_set.union(tmp[i], tmp[i+1])
    out = mf_set.get()
    return out


@task(returns=1)
def merge_core_points(chunks, comb, *args):
    tmp = [max(i) for i in list(zip(*args))]
    return tmp[chunks[comb][0]: chunks[comb][1]]

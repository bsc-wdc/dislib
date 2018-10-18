from collections import defaultdict
import constants
from pycompss.api.task import task
from pycompss.api.parameter import FILE_OUT


def orquestrate_sync_clusters(data, adj_mat, epsilon, coord, neigh_sq_loc,
                              len_neighs, quocient, res, fut_list, TH_2,
                              count_tasks, *args):
    if (len_neighs/quocient) > TH_2:
        [fut_list,
         count_tasks] = orquestrate_sync_clusters(data, adj_mat, epsilon, coord,
                                                  neigh_sq_loc, len_neighs,
                                                  quocient*2, res*2 + 0, fut_list,
                                                  TH_2, count_tasks, *args)
        [fut_list,
         count_tasks] = orquestrate_sync_clusters(data, adj_mat, epsilon, coord,
                                                  neigh_sq_loc, len_neighs,
                                                  quocient*2, res*2 + 1, fut_list,
                                                  TH_2, count_tasks, *args)
    else:
        count_tasks += 1
        fut_list.append(sync_clusters(data, adj_mat, epsilon, coord,
                                      neigh_sq_loc, quocient, res, len_neighs,
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
def sync_task(coord, cluster_labels, core_points, neigh_sq_id, *labels_versions):
    out = defaultdict(set)
    for num_label, label in enumerate(cluster_labels):
        if core_points[num_label] == constants.CORE_POINT:
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
            f_out.write(str(points[thres+num])+" "
                        + str(cluster_labels[num]) + "\n")
        f_out.close()

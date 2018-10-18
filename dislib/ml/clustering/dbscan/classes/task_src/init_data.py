from pandas import read_csv
from collections import defaultdict
import os
import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN


def count_lines(tupla, file_id, is_mn):
    if is_mn:
        path = "/gpfs/projects/bsc19/COMPSs_DATASETS/dbscan/"+str(file_id)
    else:
        path = "~/DBSCAN/data/"+str(file_id)
    path = os.path.expanduser(path)
    tmp_string = path+"/"+str(tupla[0])
    for num, j in enumerate(tupla):
        if num > 0:
            tmp_string += "_"+str(j)
    tmp_string += ".txt"
    with open(tmp_string) as infile:
        for i, line in enumerate(infile):
            pass
        return i+1


def orquestrate_init_data(tupla, file_id, len_data, quocient,
                          res, fut_list, TH_1, is_mn,
                          count_tasks):
    if (len_data/quocient) > TH_1:
        [fut_list,
         count_tasks] = orquestrate_init_data(tupla, file_id, len_data,
                                              quocient*2, res*2 + 0, fut_list,
                                              TH_1, is_mn, count_tasks)
        [fut_list,
         count_tasks] = orquestrate_init_data(tupla, file_id, len_data,
                                              quocient*2, res*2 + 1, fut_list,
                                              TH_1, is_mn, count_tasks)
    else:
        count_tasks += 1
        if is_mn:
            path = "/gpfs/projects/bsc19/COMPSs_DATASETS/dbscan/"+str(file_id)
        else:
            path = "~/DBSCAN/data/"+str(file_id)
        path = os.path.expanduser(path)
        tmp_string = path+"/"+str(tupla[0])
        for num, j in enumerate(tupla):
            if num > 0:
                tmp_string += "_"+str(j)
        tmp_string += ".txt"
        tmp_f = init_data(tmp_string, quocient, res, is_mn)
        fut_list.append(tmp_f)
    return fut_list, count_tasks


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

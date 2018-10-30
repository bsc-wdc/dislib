from collections import defaultdict
from ast import literal_eval
import itertools
import time
import argparse
from pycompss.api.api import compss_wait_on
from pycompss.api.api import compss_barrier
from dislib.cluster.dbscan import constants
from dislib.cluster.dbscan.classes import DisjointSet
from dislib.cluster.dbscan.classes import Square


def sync_relations(cluster_rules):
    out = defaultdict(set)
    for comb in cluster_rules:
        for key in cluster_rules[comb]:
            out[key] |= cluster_rules[comb][key]
    mf_set = DisjointSet(out.keys())
    for key in out:
        tmp = list(out[key])
        for i in range(len(tmp)-1):
            mf_set.union(tmp[i], tmp[i+1])
    return mf_set.get()


# DBSCAN Algorithm
def DBSCAN(eps, min_points, datafile, is_mn, print_times):
    initial_time = time.time()

    # This threshold determines the granularity of the tasks
    if is_mn:
        TH_1 = 11000
    else:
        TH_1 = 100

    # Initial Definitions (necessary?)
    dataset_info = "dataset.txt"
    count_tasks = 0

    # Data inisialitation
    dimensions = []
    f = open(dataset_info, "r")
    for line in f:
        split_line = line.split()
        if int(split_line[0]) == datafile:
            dimensions = literal_eval(split_line[1])
            break
    dimension_perms = [range(i) for i in dimensions]
    dataset = defaultdict()
    links = defaultdict()

    # For each square in the grid
    for comb in itertools.product(*dimension_perms):
        # Initialise the square object
        dataset[comb] = Square(comb, eps, dimensions)
        # Load the data to it
        count_tasks = dataset[comb].init_data(datafile, is_mn, TH_1, count_tasks)
        # Perform a local cluster
        count_tasks = dataset[comb].partial_scan(min_points, TH_1, count_tasks)

    if print_times:
        compss_barrier()
        print("Partial Scan Tasks Finished")
        ps_time = time.time() - initial_time
        print("PS Lasted: "+str(ps_time))
        cp_count = 0
        for comb in itertools.product(*dimension_perms):
            dataset[comb].core_points = compss_wait_on(dataset[comb].core_points)
            cp_count += dataset[comb].core_points.count(constants.CORE_POINT)
        print("Number of core points found: "+str(cp_count))

    # In spite of not needing a synchronization we loop again since the first
    # loop initialises all the future objects that are used afterwards
    for comb in itertools.product(*dimension_perms):
        # We retrieve all the neighbour square ids from the current square
        neigh_sq_id = dataset[comb].neigh_sq_id
        labels_versions = []
        for neigh_comb in neigh_sq_id:
            # We obtain the labels found for our points by our neighbours.
            labels_versions.append(dataset[neigh_comb].cluster_labels[comb])
        # We merge all the different labels found and return merging rules
        links[comb] = dataset[comb].sync_labels(*labels_versions)
        count_tasks += 1

    # We synchronize all the merging loops
    for comb in itertools.product(*dimension_perms):
        links[comb] = compss_wait_on(links[comb])
        count_tasks += 1

    # We sync all the links locally and broadcast the updated global labels
    # to all the workers
    updated_links = sync_relations(links)

    # We lastly output the results to text files. For performance testing
    # the following two lines could be commented.
    for comb in itertools.product(*dimension_perms):
        dataset[comb].update_labels(updated_links, is_mn, datafile)
    compss_barrier()

    print("Total number of tasks scheduled: "+str(count_tasks))
    print("Number of clusters found: "+str(len(updated_links)))
    print("Time elapsed: " + str(time.time()-initial_time))

    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DBSCAN Clustering Algorithm implemented within the PyCOMPSs'
                                                 ' framework. For a detailed guide on the usage see the '
                                                 'user guide provided.')
    parser.add_argument('epsilon', type=float, help='Radius that defines the maximum distance under which neighbors '
                                                    'are looked for.')
    parser.add_argument('min_points', type=int, help='Minimum number of neighbors for a point to '
                                                     'be considered core point.')
    parser.add_argument('datafile', type=int, help='Numeric identifier for the dataset to be used. For further '
                                                   'information see the user guide provided.')
    parser.add_argument('--is_mn', action='store_true', help='If set to true, this tells the algorithm that you are '
                                                             'running the code in the MN cluster, setting the correct '
                                                             'paths to the data files and setting the correct '
                                                             'parameters. Otherwise it assumes you are running the '
                                                             'code locally.')
    parser.add_argument('--print_times', action='store_true', help='If set to true, the timing for each task will be '
                                                                   'printed through the standard output. NOTE THAT '
                                                                   'THIS WILL LEAD TO EXTRA BARRIERS INTRODUCED IN THE'
                                                                   ' CODE. Otherwise only the total time elapsed '
                                                                   'is printed.')
    args = parser.parse_args()
    DBSCAN(**vars(args))

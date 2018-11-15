from collections import defaultdict

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task

from dislib.cluster.dbscan.classes import DisjointSet
from dislib.cluster.dbscan.classes import Square


class DBSCAN():

    def __init__(self, eps=0.5, min_points=5, grid_dim=10, grid_data=False):
        assert grid_dim >= 1, "Grid dimensions must be greater than 1."

        self._eps = eps
        self._min_points = min_points
        self._grid_dim = grid_dim
        self._grid_data = grid_data
        self.labels_ = np.empty(0)
        self._part_sizes = []
        self._sorting = []

    def fit(self, data):
        n_features = compss_wait_on(_get_n_features(data[0]))
        grid = np.empty([self._grid_dim] * n_features, dtype=object)

        assert (not self._grid_data or grid.size == len(data)), \
            "%s partitions required for grid dimension = %s and number of " \
            "features = %s. Got %s partitions instead" % (grid.size,
                                                          self._grid_dim,
                                                          n_features, len(data))

        mn, mx = self._get_min_max(data)
        bins, region_sizes = self._generate_bins(mn, mx, n_features)

        if not self._grid_data:
            self._set_part_sizes(data)
            sorted_data = self._sort_data(data, grid, bins)
        else:
            sorted_data = data

        # This threshold determines the granularity of the tasks
        TH_1 = 1000

        links = defaultdict()

        for ind in np.ndindex(grid.shape):
            grid[ind] = Square(ind, self._eps, grid.shape, region_sizes)
            grid[ind].init_data(sorted_data, grid.shape)
            grid[ind].partial_scan(self._min_points, TH_1)

        # In spite of not needing a synchronization we loop again since the first
        # loop initialises all the future objects that are used afterwards
        for ind in np.ndindex(grid.shape):
            # We retrieve all the neighbour square ids from the current square
            neigh_sq_id = grid[ind].neigh_sq_id
            labels_versions = []
            for neigh_comb in neigh_sq_id:
                # We obtain the labels found for our points by our neighbours.
                labels_versions.append(grid[neigh_comb].cluster_labels[ind])

            # We merge all the different labels found and return merging rules
            links[ind] = grid[ind].sync_labels(*labels_versions)

        # We synchronize all the merging loops
        for ind in np.ndindex(grid.shape):
            links[ind] = compss_wait_on(links[ind])

        # We sync all the links locally and broadcast the updated global labels
        # to all the workers
        updated_links = self._sync_relations(links)

        # We lastly output the results to text files. For performance testing
        # the following two lines could be commented.
        for ind in np.ndindex(grid.shape):
            grid[ind].update_labels(updated_links)
            labels = np.array(grid[ind].get_labels())
            self.labels_ = np.concatenate((self.labels_, labels))

        sorting_ind = self._get_sorting_indices()

        self.labels_ = self.labels_[np.argsort(sorting_ind)]

    def _sync_relations(self, cluster_rules):
        out = defaultdict(set)
        for comb in cluster_rules:
            for key in cluster_rules[comb]:
                out[key] |= cluster_rules[comb][key]

        mf_set = DisjointSet(out.keys())
        for key in out:
            tmp = list(out[key])
            for i in range(len(tmp) - 1):
                # Added this for loop to compare all vs all
                mf_set.union(tmp[i], tmp[i + 1])
                # for j in range(i, len(tmp)):
                #     mf_set.union(tmp[i], tmp[j])

        return mf_set.get()

    def _get_min_max(self, data):
        minmax = []

        for part in data:
            minmax.append(_min_max(part))

        minmax = compss_wait_on(minmax)
        return np.min(minmax, axis=0)[0], np.max(minmax, axis=0)[1]

    def _set_part_sizes(self, data):
        for part in data:
            self._part_sizes.append(_get_part_size(part))

    def _sort_data(self, data, grid, bins):
        sorted_data = []

        for ind in np.ndindex(grid.shape):
            vec_list = []

            # for each partition get the vectors in a particular region
            for part_ind, part in enumerate(data):
                indices, vecs = _filter(part, ind, bins)
                vec_list.append(vecs)
                self._sorting.append((part_ind, indices))

            # create a new partition representing the grid region
            sorted_data.append(_merge(*vec_list))

        return sorted_data

    def _generate_bins(self, mn, mx, n_features):
        bins = []
        region_sizes = []

        # create bins for the different regions in the grid in every dimension
        for i in range(n_features):
            # Add up a small delta to the max to include it in the binarization
            delta = mx[i] / 1e8
            bin = np.linspace(mn[i], mx[i] + delta, self._grid_dim + 1)
            bins.append(bin)
            region_sizes.append(np.max(bin[1:] - bin[0:-1]))

        return bins, np.array(region_sizes)

    def _get_sorting_indices(self):
        indices = []
        self._part_sizes = compss_wait_on(self._part_sizes)

        for part_ind, vec_ind in self._sorting:
            vec_ind = compss_wait_on(vec_ind)
            offset = np.sum(self._part_sizes[:part_ind])

            for ind in vec_ind:
                indices.append(ind + offset)

        return np.array(indices, dtype=int)


@task(returns=int)
def _get_part_size(part):
    return part.vectors.shape[0]


@task(returns=1)
def _merge(*vec_list):
    from dislib.data import Dataset
    return Dataset(np.vstack(vec_list))


@task(returns=2)
def _filter(part, ind, bins):
    filtered_vecs = part.vectors
    final_ind = np.array(range(filtered_vecs.shape[0]))

    # filter vectors by checking if they lie in the given region (specified
    # by ind)
    for col_ind in range(filtered_vecs.shape[1]):
        col = filtered_vecs[:, col_ind]
        indices = np.digitize(col, bins[col_ind]) - 1
        mask = (indices == ind[col_ind])
        filtered_vecs = filtered_vecs[mask]
        final_ind = final_ind[mask]

        if filtered_vecs.size == 0:
            break

    return final_ind, filtered_vecs


@task(returns=int)
def _get_n_features(part):
    return part.vectors.shape[1]


@task(returns=np.array)
def _min_max(part):
    mn = np.min(part.vectors, axis=0)
    mx = np.max(part.vectors, axis=0)
    return np.array([mn, mx])


@task(returns=np.array)
def _get_sorting_indices(sorting, part_sizes):
    indices = []

    for part_ind, vec_ind in sorting:
        offset = np.sum(part_sizes[:part_ind])

        for ind in vec_ind:
            indices.append(ind + offset)

    return np.array(indices)

    # if __name__ == "__main__":
    #     parser = argparse.ArgumentParser(description='DBSCAN Clustering Algorithm implemented within the PyCOMPSs'
    #                                                  ' framework. For a detailed guide on the usage see the '
    #                                                  'user guide provided.')
    #     parser.add_argument('epsilon', type=float, help='Radius that defines the maximum distance under which neighbors '
    #                                                     'are looked for.')
    #     parser.add_argument('min_points', type=int, help='Minimum number of neighbors for a point to '
    #                                                      'be considered core point.')
    #     parser.add_argument('datafile', type=int, help='Numeric identifier for the dataset to be used. For further '
    #                                                    'information see the user guide provided.')
    #     parser.add_argument('--is_mn', action='store_true', help='If set to true, this tells the algorithm that you are '
    #                                                              'running the code in the MN cluster, setting the correct '
    #                                                              'paths to the data files and setting the correct '
    #                                                              'parameters. Otherwise it assumes you are running the '
    #                                                              'code locally.')
    #     parser.add_argument('--print_times', action='store_true', help='If set to true, the timing for each task will be '
    #                                                                    'printed through the standard output. NOTE THAT '
    #                                                                    'THIS WILL LEAD TO EXTRA BARRIERS INTRODUCED IN THE'
    #                                                                    ' CODE. Otherwise only the total time elapsed '
    #                                                                    'is printed.')
    #     args = parser.parse_args()
    #     DBSCAN(**vars(args))

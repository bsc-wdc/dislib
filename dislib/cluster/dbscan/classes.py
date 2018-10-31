from itertools import product

from dislib.cluster.dbscan.tasks import *


class Square(object):

    def __init__(self, coord, epsilon, dimensions):
        self.coord = coord
        self.epsilon = epsilon
        self.dimensions = dimensions
        self.len_tot = 0
        self.offset = defaultdict()
        self.len = defaultdict()
        self.__neigh_squares_query()

    def __neigh_squares_query(self):
        dim = len(self.coord)
        neigh_squares = []
        border_squares = [int(min(max(self.epsilon * i, 1), i - 1)) for i in
                          self.dimensions]
        perm = []
        for i in range(dim):
            perm.append(range(-border_squares[i], border_squares[i] + 1))
        for comb in product(*perm):
            current = []
            for i in range(dim):
                if self.coord[i] + comb[i] in range(self.dimensions[i]):
                    current.append(self.coord[i] + comb[i])
            if len(current) == dim and current != list(self.coord):
                neigh_squares.append(tuple(current))
        neigh_squares.append(tuple(self.coord))
        self.neigh_sq_id = tuple(neigh_squares)

    def init_data(self, file_id, is_mn, TH_1):
        fut_list = []
        prev = 0
        for comb in self.neigh_sq_id:
            self.offset[comb] = prev
            self.len[comb] = count_lines(comb, file_id, is_mn)
            prev += self.len[comb]
            self.len_tot += self.len[comb]
            fut_list = orquestrate_init_data(comb, file_id,
                                             self.len[comb], 1,
                                             0, fut_list, TH_1,
                                             is_mn)
        self.points = merge_task_init_data(*fut_list)
        self.__set_neigh_thres()

    def __set_neigh_thres(self):
        out = defaultdict(list)
        for comb in self.neigh_sq_id:
            out[comb] = [self.offset[comb], self.offset[comb] + self.len[comb]]
        self.neigh_thres = out

    def partial_scan(self, min_points, TH_1, count_tasks):
        [fut_list_0,
         fut_list_1,
         fut_list_2,
         count_tasks] = orq_scan_merge(self.points, self.epsilon, min_points,
                                       TH_1, count_tasks, 1, 0, [[], [], []],
                                       self.len_tot)
        count_tasks += 2
        self.relations = merge_relations(*fut_list_1)
        self.cluster_labels = defaultdict(list)
        for comb in self.neigh_sq_id:
            count_tasks += 1
            self.cluster_labels[comb] = merge_cluster_labels(self.relations,
                                                             comb,
                                                             self.neigh_thres[
                                                                 comb],
                                                             *fut_list_0)
        self.core_points = merge_core_points(self.neigh_thres, self.coord,
                                             *fut_list_2)
        return count_tasks

    def sync_labels(self, *labels_versions):
        return sync_task(self.coord, self.cluster_labels[self.coord],
                         self.core_points, self.neigh_sq_id, *labels_versions)

    def update_labels(self, updated_relations, is_mn, file_id):
        if is_mn:
            path = "/gpfs/projects/bsc19/COMPSs_DATASETS/dbscan2/" + str(
                file_id)
        else:
            path = "~/DBSCAN/data/" + str(file_id)
        path = os.path.expanduser(path)
        tmp_string = path + "/" + str(self.coord[0])
        for num, j in enumerate(self.coord):
            if num > 0:
                tmp_string += "_" + str(j)
        tmp_string += "_OUT.txt"
        update_task(self.cluster_labels[self.coord], self.coord, self.points,
                    self.neigh_thres[self.coord][0], updated_relations,
                    tmp_string)


class Data(object):
    def __init__(self):
        self.value = []


class DisjointSet:
    _disjoint_set = list()

    #    def __init__(self, init_arr):
    #        self._disjoint_set = []
    #        if init_arr:
    #            for item in list(set(init_arr)):
    #                self._disjoint_set.append([item])

    # Alternative __init__:
    def __init__(self, init_arr):
        self._disjoint_set = []
        if init_arr:
            for item in list(init_arr):
                self._disjoint_set.append([item])

    def _find_index(self, elem):
        for item in self._disjoint_set:
            if elem in item:
                return self._disjoint_set.index(item)
        return None

    def find(self, elem):
        for item in self._disjoint_set:
            if elem in item:
                return self._disjoint_set[self._disjoint_set.index(item)]
        return None

    def union(self, elem1, elem2):
        index_elem1 = self._find_index(elem1)
        index_elem2 = self._find_index(elem2)
        if index_elem1 != index_elem2 and index_elem1 is not None and index_elem2 is not None:
            self._disjoint_set[index_elem2] = self._disjoint_set[index_elem2] + \
                                              self._disjoint_set[index_elem1]
            del self._disjoint_set[index_elem1]
        return self._disjoint_set

    def get(self):
        return self._disjoint_set

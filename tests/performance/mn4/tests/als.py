from math import ceil

import performance

import dislib as ds
from dislib.recommendation import ALS


def main():
    num_subsets = 192
    data = "/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/recommendation" \
           "/netflix/netflix_data_libsvm.txt"
    n_factors = 100

    subset_size = int(ceil(17770 / num_subsets))
    x, y = ds.load_svmlight_file(data, block_size=(subset_size, subset_size),
                                 n_features=480189, store_sparse=True)

    als = ALS(tol=0.0001, random_state=676, n_f=n_factors, max_iter=10,
              verbose=False)

    performance.measure("ALS", "Netflix", als, x, y)


if __name__ == '__main__':
    main()

from math import ceil

import performance

import dislib as ds
from dislib.recommendation import ALS


def main():
    n_blocks = 384
    data = "/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/recommendation" \
           "/netflix/netflix_data_libsvm.txt"
    n_factors = 100
    n_features = 480189

    block_size = (int(ceil(17770 / n_blocks)),
                  int(ceil(n_features / n_blocks)))

    x, y = ds.load_svmlight_file(data, block_size=block_size,
                                 n_features=n_features, store_sparse=True)

    als = ALS(tol=0.0001, random_state=676, n_f=n_factors, max_iter=10,
              verbose=False)

    performance.measure("ALS", "Netflix", als.fit, x)


if __name__ == '__main__':
    main()

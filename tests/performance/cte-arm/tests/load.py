from math import ceil

import performance

import dislib as ds


def main():
    data = "/fefs/scratch/bsc19/bsc19029/PERFORMANCE/datasets/" \
           "netflix_data_libsvm.txt"
    n_blocks = 384
    n_features = 480189
    n_samples = 17770

    block_size = (int(ceil(n_samples / n_blocks)),
                  int(ceil(n_features / n_blocks)))

    performance.measure("Load", "Netflix", ds.load_svmlight_file, data,
                        block_size=block_size, n_features=n_features,
                        store_sparse=True)


if __name__ == "__main__":
    main()

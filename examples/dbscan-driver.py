import argparse
import os
import time

import numpy as np
from pycompss.api.api import compss_barrier

from dislib.cluster import DBSCAN
from dislib.data import (load_libsvm_file, load_libsvm_files, load_txt_file,
                         load_txt_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libsvm", help="read files in libsvm format",
                        action="store_true")
    parser.add_argument("-dt", "--detailed_times",
                        help="get detailed execution times (read and fit)",
                        action="store_true")
    parser.add_argument("-e", "--epsilon", metavar="EPSILON", type=float,
                        help="default is 0.5", default=0.5)
    parser.add_argument("-r", "--regions", metavar="N_REGIONS", type=int,
                        help="number of regions to create", default=1)
    parser.add_argument("-d", "--dimensions", metavar="DIMENSIONS", type=str,
                        help="comma separated dimensions to use in the grid",
                        required=False)
    parser.add_argument("-x", "--max_samples", metavar="MAX_SAMPLES", type=int,
                        help="maximum samples to process per task ("
                             "default is 1000)", default=1000)
    parser.add_argument("-m", "--min_samples", metavar="MIN_SAMPLES",
                        type=int, help="default is 5", default=5)
    parser.add_argument("--arrange", action="store_true",
                        help="arrange data before clustering")
    parser.add_argument("-p", "--part_size", metavar="PART_SIZE", type=int,
                        help="size of the partitions in which to divide the "
                             "input dataset (default is 100)", default=100)
    parser.add_argument("-f", "--features", metavar="N_FEATURES", type=int,
                        default=None, required=True)
    parser.add_argument("--dense", help="use dense data structures",
                        action="store_true")
    parser.add_argument("--labeled", help="dataset is labeled",
                        action="store_true")
    parser.add_argument("train_data",
                        help="File or directory containing files "
                             "(if a directory is provided PART_SIZE is "
                             "ignored)", type=str)
    args = parser.parse_args()

    train_data = args.train_data

    s_time = time.time()
    read_time = 0

    sparse = not args.dense

    label_col = None

    if args.labeled:
        label_col = "last"

    if os.path.isdir(train_data):
        if args.libsvm:
            data = load_libsvm_files(train_data, args.features,
                                     store_sparse=sparse)
        else:
            data = load_txt_files(train_data, args.features,
                                  label_col=label_col)
    else:
        if args.libsvm:
            data = load_libsvm_file(train_data, subset_size=args.part_size,
                                    n_features=args.features,
                                    store_sparse=sparse)
        else:
            data = load_txt_file(train_data, subset_size=args.part_size,
                                 n_features=args.features, label_col=label_col)

    if args.detailed_times:
        compss_barrier()
        read_time = time.time() - s_time
        s_time = time.time()

    dims = range(args.features)

    if args.dimensions:
        dims = args.dimensions.split(",")
        dims = np.array(dims, dtype=int)

    dbscan = DBSCAN(eps=args.epsilon, min_samples=args.min_samples,
                    max_samples=args.max_samples, n_regions=args.regions,
                    dimensions=dims, arrange_data=args.arrange)
    dbscan.fit(data)

    compss_barrier()
    fit_time = time.time() - s_time

    out = [dbscan._eps, dbscan._min_samples, dbscan._max_samples,
           dbscan._n_regions, len(dims), args.part_size, dbscan.n_clusters,
           read_time, fit_time]

    print(out)


if __name__ == "__main__":
    main()

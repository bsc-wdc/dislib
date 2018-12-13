import argparse
import os
import time

from pycompss.api.api import barrier

from dislib.cluster import KMeans
from dislib.data import (load_libsvm_file, load_libsvm_files, load_csv_file,
                         load_csv_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libsvm", help="read files in libsvm format",
                        action="store_true")
    parser.add_argument("-dt", "--detailed_times",
                        help="get detailed execution times (read and fit)",
                        action="store_true")
    parser.add_argument("-a", "--arity", metavar="CASCADE_ARITY", type=int,
                        help="default is 50", default=50)
    parser.add_argument("-c", "--clusters", metavar="N_CLUSTERS", type=int,
                        help="default is 2", default=2)
    parser.add_argument("-p", "--part_size", metavar="PART_SIZE", type=int,
                        help="size of the partitions in which to divide the "
                             "input dataset (default is 100)", default=100)
    parser.add_argument("-i", "--iteration", metavar="MAX_ITERATIONS",
                        type=int, help="default is 5", default=5)
    parser.add_argument("-f", "--features", metavar="N_FEATURES", type=int,
                        default=None, required=True)
    parser.add_argument("--dense", help="use dense data structures",
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

    if os.path.isdir(train_data):
        if args.libsvm:
            data = load_libsvm_files(train_data, args.features,
                                     store_sparse=sparse)
        else:
            data = load_csv_files(train_data, args.features, label_col="last")
    else:
        if args.libsvm:
            data = load_libsvm_file(train_data, subset_size=args.part_size,
                                    n_features=args.features,
                                    store_sparse=sparse)
        else:
            data = load_csv_file(train_data, subset_size=args.part_size,
                                 n_features=args.features, label_col="last")

    if args.detailed_times:
        barrier()
        read_time = time.time() - s_time
        s_time = time.time()

    kmeans = KMeans(n_clusters=args.clusters, max_iter=args.iteration,
                    arity=args.arity)
    kmeans.fit(data)

    barrier()
    fit_time = time.time() - s_time

    out = [args.kernel, args.arity, args.part_size, read_time, fit_time]

    print(out)


if __name__ == "__main__":
    main()

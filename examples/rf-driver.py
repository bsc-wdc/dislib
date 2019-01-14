import argparse
import os
import time

import numpy as np
from pycompss.api.api import barrier

from dislib.classification import RandomForestClassifier
from dislib.data import (load_libsvm_file, load_libsvm_files, load_csv_file,
                         load_csv_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libsvm", help="read files in libsvm format",
                        action="store_true")
    parser.add_argument("-dt", "--detailed_times",
                        help="get detailed execution times (read and fit)",
                        action="store_true")
    parser.add_argument("-e", "--estimators", metavar="N_ESTIMATORS",
                        type=int, help="default is 10", default=10)
    parser.add_argument("-p", "--part_size", metavar="PART_SIZE", type=int,
                        help="size of the partitions in which to divide the "
                             "input dataset (default is 100)", default=100)
    parser.add_argument("-md", "--max_depth", metavar="MAX_DEPTH",
                        type=int, help="default is np.inf", default=np.inf)
    parser.add_argument("-dd", "--dist_depth", metavar="DIST_DEPTH", type=int,
                        help="default is auto", default=-1)
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

    if args.dist_depth < 0:
        dist_depth = "auto"
    else:
        dist_depth = args.dist_depth

    forest = RandomForestClassifier(n_estimators=args.estimators,
                                    max_depth=args.max_depth,
                                    distr_depth=dist_depth)
    forest.fit(data)

    barrier()
    fit_time = time.time() - s_time

    out = [forest.n_estimators, forest.distr_depth, forest.max_depth,
           read_time, fit_time]

    print(out)


if __name__ == "__main__":
    main()

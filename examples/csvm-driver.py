import argparse
import csv
import os

import time

import numpy as np
from sklearn.datasets import load_svmlight_file

from dislib.data import *
from dislib.classification import CascadeSVM

from pycompss.api.api import barrier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libsvm", help="read files in libsvm format",
                        action="store_true")
    parser.add_argument("-dt", "--detailed_times",
                        help="get detailed execution times (read and fit)",
                        action="store_true")
    parser.add_argument("-k", metavar="KERNEL", type=str,
                        help="linear or rbf (default is rbf)",
                        choices=["linear", "rbf"], default="rbf")
    parser.add_argument("-a", metavar="CASCADE_ARITY", type=int,
                        help="default is 2", default=2)
    parser.add_argument("-p", metavar="PART_SIZE", type=int,
                        help="size of the partitions in which to divide the "
                             "input dataset")
    parser.add_argument("-i", metavar="MAX_ITERATIONS", type=int,
                        help="default is 5", default=5)
    parser.add_argument("-g", metavar="GAMMA", type=float,
                        help="(only for rbf kernel) default is 1 / n_features",
                        default=None)
    parser.add_argument("-c", metavar="C", type=float, help="default is 1",
                        default=1)
    parser.add_argument("-f", metavar="N_FEATURES", type=int,
                        help="mandatory if --libsvm option is used and "
                             "train_data is a directory (optional otherwise)",
                        default=None)
    parser.add_argument("-t", metavar="TEST_FILE_PATH",
                        help="test CSV file path", type=str, required=False)
    parser.add_argument("-o", metavar="OUTPUT_FILE_PATH",
                        help="output file path", type=str, required=False)
    parser.add_argument("-nd", metavar="N_DATASETS", type=int,
                        help="number of times to load the dataset", default=1)
    parser.add_argument("--convergence", help="check for convergence",
                        action="store_true")
    parser.add_argument("--dense", help="use dense data structures",
                        action="store_true")
    parser.add_argument("train_data",
                        help="File or directory containing files "
                             "(if a directory is provided PART_SIZE is "
                             "ignored)",
                        type=str)
    args = parser.parse_args()

    train_data = args.train_data

    if not args.g:
        gamma = "auto"
    else:
        gamma = args.g

    data = []

    s_time = time.time()

    if os.path.isdir(train_data):
        _loader_func = load_libsvm_files if args.libsvm else load_csv_files
        for _ in range(args.nd):
            data.append(_loader_func(train_data, args.f))
    else:
        _loader_func = load_libsvm_file if args.libsvm else load_csv_file
        for _ in range(args.nd):
            data.append(_loader_func(train_data, args.p, args.f))

    if args.detailed_times:
        barrier()

    csvm = CascadeSVM(cascade_arity=args.a, max_iter=args.i, c=args.c,
                      gamma=gamma, check_convergence=args.convergence)

    for d in data:
        csvm.fit(d)

    out = [args.k, args.a, args.p, csvm._clf_params["gamma"], args.c,
           csvm.iterations, csvm.converged]

    if os.path.isdir(train_data):
        n_files = os.listdir(train_data)
        out.append(len(n_files))

    if args.t:
        if args.libsvm:
            testx, testy = load_svmlight_file(args.t, args.f)

            if args.dense:
                testx = testx.toarray()

            out.append(csvm.score(testx, testy))
        else:
            test = np.loadtxt(args.t, delimiter=",", dtype=float)
            out.append(csvm.score(test[:, :-1], test[:, -1]))

    if args.o:
        with open(args.o, "ab") as f:
            wr = csv.writer(f)
            wr.writerow(out)
    else:
        print(out)


if __name__ == "__main__":
    main()

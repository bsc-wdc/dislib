import argparse
import csv
import os
import time

from pycompss.api.api import barrier

from dislib.classification import CascadeSVM
from dislib.data import (load_libsvm_file, load_libsvm_files, load_txt_file,
                         load_txt_files)
from dislib.utils import shuffle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libsvm", help="read files in libsvm format",
                        action="store_true")
    parser.add_argument("-dt", "--detailed_times",
                        help="get detailed execution times (read and fit)",
                        action="store_true")
    parser.add_argument("-k", "--kernel", metavar="KERNEL", type=str,
                        help="linear or rbf (default is rbf)",
                        choices=["linear", "rbf"], default="rbf")
    parser.add_argument("-a", "--arity", metavar="CASCADE_ARITY", type=int,
                        help="default is 2", default=2)
    parser.add_argument("-p", "--part_size", metavar="PART_SIZE", type=int,
                        help="size of the partitions in which to divide the "
                             "input dataset (default is 100)", default=100)
    parser.add_argument("-i", "--iteration", metavar="MAX_ITERATIONS",
                        type=int, help="default is 5", default=5)
    parser.add_argument("-g", "--gamma", metavar="GAMMA", type=float,
                        help="(only for rbf kernel) default is 1 / n_features",
                        default=None)
    parser.add_argument("-c", metavar="C", type=float, default=1,
                        help="Penalty parameter C of the error term. "
                             "Default:1")
    parser.add_argument("-f", "--features", metavar="N_FEATURES", type=int,
                        default=None, required=True)
    parser.add_argument("-t", "--test-file", metavar="TEST_FILE_PATH",
                        help="test CSV file path", type=str, required=False)
    parser.add_argument("-o", "--output_file", metavar="OUTPUT_FILE_PATH",
                        help="output file path", type=str, required=False)
    parser.add_argument("-nd", "--n_datasets", metavar="N_DATASETS", type=int,
                        help="number of times to load the dataset", default=1)
    parser.add_argument("--convergence", help="check for convergence",
                        action="store_true")
    parser.add_argument("--dense", help="use dense data structures",
                        action="store_true")
    parser.add_argument("train_data",
                        help="File or directory containing files "
                             "(if a directory is provided PART_SIZE is "
                             "ignored)", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--shuffle", help="shuffle input data",
                        action="store_true")
    args = parser.parse_args()

    train_data = args.train_data

    s_time = time.time()
    read_time = 0

    if not args.gamma:
        gamma = "auto"
    else:
        gamma = args.gamma

    data = []
    sparse = not args.dense

    if os.path.isdir(train_data):
        if args.libsvm:
            for _ in range(args.n_datasets):
                data.append(load_libsvm_files(train_data, args.features,
                                              store_sparse=sparse))
        else:
            for _ in range(args.n_datasets):
                data.append(load_txt_files(train_data, args.features,
                                           label_col="last"))
    else:
        if args.libsvm:
            for _ in range(args.n_datasets):
                data.append(
                    load_libsvm_file(train_data, subset_size=args.part_size,
                                     n_features=args.features,
                                     store_sparse=sparse))
        else:
            for _ in range(args.n_datasets):
                data.append(
                    load_txt_file(train_data, subset_size=args.part_size,
                                  n_features=args.features, label_col="last"))

    if args.detailed_times:
        barrier()
        read_time = time.time() - s_time
        s_time = time.time()

    csvm = CascadeSVM(cascade_arity=args.arity, max_iter=args.iteration,
                      c=args.c, gamma=gamma,
                      check_convergence=args.convergence, verbose=args.verbose)

    for d in data:
        dataset = d

        if args.shuffle:
            dataset = shuffle(d)

        csvm.fit(dataset)

    barrier()
    fit_time = time.time() - s_time

    out = [args.kernel, args.arity, args.part_size, csvm._clf_params["gamma"],
           args.c, csvm.iterations, csvm.converged, read_time, fit_time]

    if os.path.isdir(train_data):
        n_files = os.listdir(train_data)
        out.append(len(n_files))

    if args.test_file:
        if args.libsvm:
            test_data = load_libsvm_file(args.test_file,
                                         n_features=args.features,
                                         subset_size=args.part_size,
                                         store_sparse=sparse)
        else:
            test_data = load_txt_file(args.test_file,
                                      n_features=args.features,
                                      subset_size=args.part_size,
                                      label_col="last")

        out.append(csvm.score(test_data))

    if args.output_file:
        with open(args.output_file, "ab") as f:
            wr = csv.writer(f)
            wr.writerow(out)
    else:
        print(out)


if __name__ == "__main__":
    main()

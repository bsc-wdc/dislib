import argparse
import csv
import os
import time

from pycompss.api.api import barrier, compss_wait_on

import dislib as ds
from dislib.classification import CascadeSVM
from dislib.utils import shuffle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svmlight", help="read files in SVMLight format",
                        action="store_true")
    parser.add_argument("-dt", "--detailed_times",
                        help="get detailed execution times (read and fit)",
                        action="store_true")
    parser.add_argument("-k", "--kernel", metavar="KERNEL", type=str,
                        help="linear or rbf (default is rbf)",
                        choices=["linear", "rbf"], default="rbf")
    parser.add_argument("-a", "--arity", metavar="CASCADE_ARITY", type=int,
                        help="default is 2", default=2)
    parser.add_argument("-b", "--block_size", metavar="BLOCK_SIZE", type=str,
                        help="two comma separated ints that represent the "
                             "size of the blocks in which to divide the input "
                             "data (default is 100,100)",
                        default="100,100")
    parser.add_argument("-i", "--iteration", metavar="MAX_ITERATIONS",
                        type=int, help="default is 5", default=5)
    parser.add_argument("-g", "--gamma", metavar="GAMMA", type=float,
                        help="(only for rbf kernel) default is 1 / n_features",
                        default=None)
    parser.add_argument("-c", metavar="C", type=float, default=1,
                        help="Penalty parameter C of the error term. "
                             "Default:1")
    parser.add_argument("-f", "--features", metavar="N_FEATURES",
                        help="number of features of the input data "
                             "(only for SVMLight files)",
                        type=int, default=None, required=False)
    parser.add_argument("-t", "--test-file", metavar="TEST_FILE_PATH",
                        help="test file path", type=str, required=False)
    parser.add_argument("-o", "--output_file", metavar="OUTPUT_FILE_PATH",
                        help="output file path", type=str, required=False)
    parser.add_argument("--convergence", help="check for convergence",
                        action="store_true")
    parser.add_argument("--dense", help="store data in dense format (only "
                                        "for SVMLight files)",
                        action="store_true")
    parser.add_argument("train_data",
                        help="input file in CSV or SVMLight format", type=str)
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

    sparse = not args.dense

    bsize = args.block_size.split(",")
    block_size = (int(bsize[0]), int(bsize[1]))

    if args.svmlight:
        x, y = ds.load_svmlight_file(train_data, block_size, args.features,
                                     sparse)
    else:
        x = ds.load_txt_file(train_data, block_size)
        y = x[:, x.shape[1] - 2: x.shape[1] - 1]
        x = x[:, :x.shape[1] - 1]

    if args.shuffle:
        x, y = shuffle(x, y)

    if args.detailed_times:
        barrier()
        read_time = time.time() - s_time
        s_time = time.time()

    csvm = CascadeSVM(cascade_arity=args.arity, max_iter=args.iteration,
                      c=args.c, gamma=gamma,
                      check_convergence=args.convergence, verbose=args.verbose)

    csvm.fit(x, y)

    barrier()
    fit_time = time.time() - s_time

    out = [args.kernel, args.arity, args.part_size, csvm._clf_params["gamma"],
           args.c, csvm.iterations, csvm.converged, read_time, fit_time]

    if os.path.isdir(train_data):
        n_files = os.listdir(train_data)
        out.append(len(n_files))

    if args.test_file:
        if args.svmlight:
            x_test, y_test = ds.load_svmlight_file(args.test_file, block_size,
                                                   args.features,
                                                   sparse)
        else:
            x_test = ds.load_txt_file(args.test_file, block_size)
            y_test = x_test[:, x_test.shape[1] - 1: x_test.shape[1]]
            x_test = x_test[:, :x_test.shape[1] - 1]

        out.append(compss_wait_on(csvm.score(x_test, y_test)))

    if args.output_file:
        with open(args.output_file, "ab") as f:
            wr = csv.writer(f)
            wr.writerow(out)
    else:
        print(out)


if __name__ == "__main__":
    main()

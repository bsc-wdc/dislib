import argparse
import time

import numpy as np
from pycompss.api.api import barrier, compss_wait_on

import dislib as ds
from dislib.classification import RandomForestClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svmlight", help="read files in SVMLight format",
                        action="store_true")
    parser.add_argument("-dt", "--detailed_times",
                        help="get detailed execution times (read and fit)",
                        action="store_true")
    parser.add_argument("-e", "--estimators", metavar="N_ESTIMATORS",
                        type=int, help="default is 10", default=10)
    parser.add_argument("-b", "--block_size", metavar="BLOCK_SIZE", type=str,
                        help="two comma separated ints that represent the "
                             "size of the blocks in which to divide the input "
                             "data (default is 100,100)",
                        default="100,100")
    parser.add_argument("-md", "--max_depth", metavar="MAX_DEPTH",
                        type=int, help="default is np.inf", required=False)
    parser.add_argument("-dd", "--dist_depth", metavar="DIST_DEPTH", type=int,
                        help="default is auto", required=False)
    parser.add_argument("-f", "--features", metavar="N_FEATURES",
                        help="number of features of the input data "
                             "(only for SVMLight files)",
                        type=int, default=None, required=False)
    parser.add_argument("--dense", help="use dense data structures",
                        action="store_true")
    parser.add_argument("-t", "--test-file", metavar="TEST_FILE_PATH",
                        help="test file path", type=str, required=False)
    parser.add_argument("train_data",
                        help="input file in CSV or SVMLight format", type=str)
    args = parser.parse_args()

    train_data = args.train_data

    s_time = time.time()
    read_time = 0

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

    if args.detailed_times:
        barrier()
        read_time = time.time() - s_time
        s_time = time.time()

    if args.dist_depth:
        dist_depth = args.dist_depth
    else:
        dist_depth = "auto"

    if args.max_depth:
        max_depth = args.max_depth
    else:
        max_depth = np.inf

    forest = RandomForestClassifier(n_estimators=args.estimators,
                                    max_depth=max_depth,
                                    distr_depth=dist_depth)
    forest.fit(x, y)

    barrier()
    fit_time = time.time() - s_time

    out = [forest.n_estimators, forest.distr_depth, forest.max_depth,
           read_time, fit_time]

    if args.test_file:
        if args.svmlight:
            x_test, y_test = ds.load_svmlight_file(args.test_file, block_size,
                                                   args.features,
                                                   sparse)
        else:
            x_test = ds.load_txt_file(args.test_file, block_size)
            y_test = x_test[:, x_test.shape[1] - 1: x_test.shape[1]]
            x_test = x_test[:, :x_test.shape[1] - 1]

        out.append(compss_wait_on(forest.score(x_test, y_test)))

    print(out)


if __name__ == "__main__":
    main()

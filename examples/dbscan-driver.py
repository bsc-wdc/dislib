import argparse
import time

import numpy as np
from pycompss.api.api import compss_barrier

import dislib as ds
from dislib.cluster import DBSCAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svmlight", help="read file in SVMlLight format",
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
    parser.add_argument("-b", "--block_size", metavar="BLOCK_SIZE", type=str,
                        help="two comma separated ints that represent the "
                             "size of the blocks in which to divide the input "
                             "data (default is 100,100)",
                        default="100,100")
    parser.add_argument("-f", "--features", metavar="N_FEATURES",
                        help="number of features of the input data "
                             "(only for SVMLight files)",
                        type=int, default=None, required=False)
    parser.add_argument("--dense", help="store data in dense format (only "
                                        "for SVMLight files)",
                        action="store_true")
    parser.add_argument("--labeled", help="the last column of the input file "
                                          "represents labels (only for text "
                                          "files)",
                        action="store_true")
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

    n_features = x.shape[1]

    if args.labeled and not args.svmlight:
        x = x[:, :n_features - 1]

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
                    dimensions=dims)
    dbscan.fit(x)

    compss_barrier()
    fit_time = time.time() - s_time

    out = [dbscan.eps, dbscan.min_samples, dbscan.max_samples,
           dbscan.n_regions, len(dims), args.part_size, dbscan.n_clusters,
           read_time, fit_time]

    print(out)


if __name__ == "__main__":
    main()

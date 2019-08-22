import argparse
import time

from pycompss.api.api import barrier

import dislib as ds
from dislib.cluster import KMeans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svmlight", help="read files in SVMLight format",
                        action="store_true")
    parser.add_argument("-dt", "--detailed_times",
                        help="get detailed execution times (read and fit)",
                        action="store_true")
    parser.add_argument("-a", "--arity", metavar="CASCADE_ARITY", type=int,
                        help="default is 50", default=50)
    parser.add_argument("-c", "--centers", metavar="N_CENTERS", type=int,
                        help="default is 2", default=2)
    parser.add_argument("-b", "--block_size", metavar="BLOCK_SIZE", type=str,
                        help="two comma separated ints that represent the "
                             "size of the blocks in which to divide the input "
                             "data (default is 100,100)",
                        default="100,100")
    parser.add_argument("-i", "--iteration", metavar="MAX_ITERATIONS",
                        type=int, help="default is 5", default=5)
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
        barrier()
        read_time = time.time() - s_time
        s_time = time.time()

    kmeans = KMeans(n_clusters=args.clusters, max_iter=args.iteration,
                    arity=args.arity, verbose=True)
    kmeans.fit(x)

    barrier()
    fit_time = time.time() - s_time

    out = [args.clusters, args.arity, args.part_size, read_time, fit_time]

    print(out)


if __name__ == "__main__":
    main()

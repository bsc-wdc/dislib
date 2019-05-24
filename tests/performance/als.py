import argparse
from math import ceil
from time import time

from dislib.data import load_libsvm_file
from dislib.recommendation import ALS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_subsets", type=int)
    parser.add_argument("--num_factors", type=int)
    parser.add_argument("--data", type=str, help='Dataset in libsvm format')

    args = parser.parse_args()

    num_subsets = args.num_subsets
    data = args.data
    n_factors = args.num_factors

    subset_size = int(ceil(17770 / num_subsets))
    ds = load_libsvm_file(data, subset_size=subset_size, n_features=480189)

    exec_start = time()

    als = ALS(tol=0.0001, random_state=676, n_f=n_factors, max_iter=10,
              verbose=True)

    als.fit(ds)
    exec_end = time()

    print("Execution time: %.2f" % (exec_end - exec_start))

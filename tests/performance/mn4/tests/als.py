import performance
from math import ceil

from dislib.data import load_libsvm_file
from dislib.recommendation import ALS


def main():
    num_subsets = 192
    data = "/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/recommendation" \
           "/netflix/netflix_data_libsvm.txt"
    n_factors = 100

    subset_size = int(ceil(17770 / num_subsets))
    ds = load_libsvm_file(data, subset_size=subset_size, n_features=480189)

    als = ALS(tol=0.0001, random_state=676, n_f=n_factors, max_iter=10,
              verbose=False)

    performance.measure("ALS", "Netflix", als, ds)


if __name__ == '__main__':
    main()

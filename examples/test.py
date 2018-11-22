# encoding utf8


import sys
from dislib.classification import CascadeSVM
from dislib.data import load_libsvm_file


def main():
    file_path = sys.argv[1]
    print("Hola!")
    csvm = CascadeSVM(gamma=0.001, c=10000, max_iter=2, check_convergence=True)
    data = load_libsvm_file(file_path, subset_size=20, n_features=780, store_sparse=True)
    csvm.fit(data)
    print("Finished")


if __name__ == "__main__":
    main()

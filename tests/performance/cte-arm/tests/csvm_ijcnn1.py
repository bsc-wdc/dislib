import performance

import dislib as ds
from dislib.classification import CascadeSVM


def main():
    x_ij, y_ij = ds.load_svmlight_file(
        "/fefs/scratch/bsc19/bsc19029/PERFORMANCE/datasets/train",
        block_size=(5000, 22), n_features=22, store_sparse=True)

    csvm = CascadeSVM(c=10000, gamma=0.01)

    performance.measure("CSVM", "ijcnn1", csvm.fit, x_ij, y_ij)


if __name__ == "__main__":
    main()

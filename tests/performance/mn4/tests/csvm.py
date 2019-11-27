import performance

import dislib as ds
from dislib.classification import CascadeSVM
from dislib.utils import shuffle


def main():
    x_kdd = ds.load_txt_file(
        "/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/kdd99/train.csv",
        block_size=(11482, 122))

    x_kdd = shuffle(x_kdd)
    y_kdd = x_kdd[:, 121:122]
    x_kdd = x_kdd[:, :121]

    x_ij, y_ij = ds.load_svmlight_file(
        "/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/ijcnn1/train",
        block_size=(5000, 22), n_features=22, store_sparse=True)

    csvm = CascadeSVM(c=10000, gamma=0.01)

    performance.measure("CSVM", "KDD99", csvm.fit, x_kdd, y_kdd)
    performance.measure("CSVM", "ijcnn1", csvm.fit, x_ij, y_ij)


if __name__ == "__main__":
    main()

import performance

from dislib.classification import CascadeSVM
from dislib.data import load_txt_files, load_libsvm_file


def main():
    kdd = load_txt_files(
        "/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/kdd99/partitions/384",
        n_features=121, label_col="last")
    ijcnn1 = load_libsvm_file(
        "/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/ijcnn1/train",
        subset_size=5000, n_features=22)

    csvm = CascadeSVM(c=10000, gamma=0.01)

    performance.measure("CSVM", "KDD99", csvm, kdd)
    performance.measure("CSVM", "ijcnn1", csvm, ijcnn1)


if __name__ == "__main__":
    main()

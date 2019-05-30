import performance

from dislib.classification import RandomForestClassifier
from dislib.data import load_txt_files, load_libsvm_file


def main():
    kdd = load_txt_files(
        "/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/kdd99/partitions/384",
        n_features=121, label_col="last")
    covtype = load_libsvm_file(
        "/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/covtype/train",
        subset_size=5000, n_features=54, store_sparse=False)

    rf = RandomForestClassifier(n_estimators=100, distr_depth=2)
    performance.measure("RF", "KDD99", rf, kdd)

    rf = RandomForestClassifier(n_estimators=100, distr_depth=2)
    performance.measure("RF", "covtype", rf, covtype)


if __name__ == "__main__":
    main()

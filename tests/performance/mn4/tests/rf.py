import performance

import dislib as ds
from dislib.classification import RandomForestClassifier


def main():
    x_kdd = ds.load_txt_file(
        "/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/kdd99/train.csv",
        block_size=(11482, 122))

    y_kdd = x_kdd[:, 121:122]
    x_kdd = x_kdd[:, :121]

    x_mn, y_mn = ds.load_svmlight_file(
        "/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/mnist/train.scaled",
        block_size=(5000, 780), n_features=780, store_sparse=False)

    rf = RandomForestClassifier(n_estimators=100, distr_depth=2)
    performance.measure("RF", "KDD99", rf.fit, x_kdd, y_kdd)

    rf = RandomForestClassifier(n_estimators=100, distr_depth=2)
    performance.measure("RF", "mnist", rf.fit, x_mn, y_mn)


if __name__ == "__main__":
    main()

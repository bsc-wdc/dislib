import performance

import dislib as ds
from dislib.classification import RandomForestClassifier


def main():

    x_mn, y_mn = ds.load_svmlight_file(
        "/fefs/scratch/bsc19/bsc19029/PERFORMANCE/datasets/train.scaled",
        block_size=(5000, 780), n_features=780, store_sparse=False)

    rf = RandomForestClassifier(n_estimators=100, distr_depth=2)
    performance.measure("RF", "mnist", rf.fit, x_mn, y_mn)


if __name__ == "__main__":
    main()

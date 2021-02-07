import performance

import dislib as ds
from dislib.classification import RandomForestClassifier


def main():
    x_kdd = ds.load_txt_file(
        "/fefs/scratch/bsc19/bsc19029/PERFORMANCE/datasets/train.csv",
        block_size=(11482, 122))

    y_kdd = x_kdd[:, 121:122]
    x_kdd = x_kdd[:, :121]

    rf = RandomForestClassifier(n_estimators=100, distr_depth=2)
    performance.measure("RF", "KDD99", rf.fit, x_kdd, y_kdd)


if __name__ == "__main__":
    main()

import performance

import dislib as ds
from dislib.regression import LinearRegression


def main():
    x_kdd = ds.load_txt_file(
        "/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/kdd99/train.csv",
        block_size=(11482, 122))

    y_kdd = x_kdd[:, 121:122]
    x_kdd = x_kdd[:, :121]

    regression = LinearRegression(arity=48)

    performance.measure("LR", "KDD99", regression.fit, x_kdd, y_kdd)


if __name__ == "__main__":
    main()

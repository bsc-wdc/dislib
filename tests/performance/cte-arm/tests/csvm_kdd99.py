import performance

import dislib as ds
from dislib.classification import CascadeSVM
from dislib.utils import shuffle


def main():
    x_kdd = ds.load_txt_file(
        "/fefs/scratch/bsc19/bsc19029/PERFORMANCE/datasets/train.csv",
        block_size=(11482, 122))

    x_kdd = shuffle(x_kdd)
    y_kdd = x_kdd[:, 121:122]
    x_kdd = x_kdd[:, :121]

    csvm = CascadeSVM(c=10000, gamma=0.01)

    performance.measure("CSVM", "KDD99", csvm.fit, x_kdd, y_kdd)


if __name__ == "__main__":
    main()

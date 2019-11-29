import performance

import dislib as ds
from dislib.decomposition import PCA


def main():
    x_kdd = ds.load_txt_file(
        "/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/kdd99/train.csv",
        block_size=(11482, 122))

    x_kdd = x_kdd[:, :121]
    pca = PCA(arity=48)

    performance.measure("PCA", "KDD99", pca.fit, x_kdd)


if __name__ == "__main__":
    main()

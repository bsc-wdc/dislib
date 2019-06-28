import performance

from dislib.cluster import DBSCAN
from dislib.data import load_txt_file


def main():
    data = load_txt_file("/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/gaia"
                         "/dbscan/data_scaled.csv", subset_size=10000,
                         n_features=5)

    dbscan = DBSCAN(eps=0.19, min_samples=5, max_samples=5000, n_regions=17,
                    dimensions=[0, 1], arrange_data=True)
    performance.measure("DBSCAN", "gaia", dbscan, data)


if __name__ == "__main__":
    main()

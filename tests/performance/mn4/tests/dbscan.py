import performance

import dislib as ds
from dislib.cluster import DBSCAN


def main():
    data = ds.load_txt_file("/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/gaia"
                            "/dbscan/data_scaled.csv", block_size=(10000, 5))

    dbscan = DBSCAN(eps=0.19, min_samples=5, max_samples=5000, n_regions=17,
                    dimensions=[0, 1])
    performance.measure("DBSCAN", "gaia", dbscan.fit, data)


if __name__ == "__main__":
    main()

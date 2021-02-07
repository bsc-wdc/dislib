import performance

import dislib as ds
from dislib.cluster import DBSCAN


def main():
    file = "/fefs/scratch/bsc19/bsc19029/PERFORMANCE/datasets/data_scaled.csv"
    data = ds.load_txt_file(file, block_size=(10000, 5))

    dbscan = DBSCAN(eps=0.19, min_samples=5, max_samples=5000, n_regions=17,
                    dimensions=[0, 1])
    performance.measure("DBSCAN", "gaia", dbscan.fit, data)


if __name__ == "__main__":
    main()

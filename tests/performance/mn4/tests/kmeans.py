import numpy as np
import performance

import dislib as ds
from dislib.cluster import KMeans


def main():
    n_samples = 3000000
    n_chunks = 384
    chunk_size = int(np.ceil(n_samples / n_chunks))
    n_features = 100
    n_clusters = 500

    x = ds.random_array((n_samples, n_features), (chunk_size, n_features))

    km = KMeans(n_clusters=n_clusters, max_iter=5, tol=0, arity=50)
    performance.measure("KMeans", "3M", km, x)


if __name__ == "__main__":
    main()

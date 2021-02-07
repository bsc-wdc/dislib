import numpy as np
import performance

import dislib as ds
from dislib.cluster import GaussianMixture


def main():
    n_samples = 100000000
    n_chunks = 768
    chunk_size = int(np.ceil(n_samples / n_chunks))
    n_features = 100
    n_clusters = 50

    x = ds.random_array((n_samples, n_features), (chunk_size, n_features))
    gmm = GaussianMixture(n_components=n_clusters, max_iter=5, tol=0,
                          init_params="random")
    performance.measure("GMM", "100M", gmm.fit, x)


if __name__ == "__main__":
    main()

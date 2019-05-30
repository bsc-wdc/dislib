import numpy as np
import performance
from pycompss.api.api import compss_barrier
from pycompss.api.task import task

from dislib.cluster import KMeans
from dislib.data import Dataset, Subset


def main():
    n_samples = 100000
    n_chunks = 384
    chunk_size = int(np.ceil(n_samples / n_chunks))
    n_features = 100
    n_clusters = 500

    x = gen_random((n_samples, n_features), chunk_size)
    compss_barrier()

    km = KMeans(n_clusters=n_clusters, max_iter=5, tol=0, arity=50)
    performance.measure("KMeans", km, x)


def gen_random(shape, subset_size):
    n_samples = shape[0]
    n_features = shape[1]
    n_subsets = int(n_samples / subset_size)

    dataset = Dataset(n_features=n_features, sparse=False)

    for i in range(n_subsets):
        dataset.append(_random_subset(subset_size, n_features))

    return dataset


@task(returns=1)
def _random_subset(size, n_features):
    return Subset(np.random.random((size, n_features)))


if __name__ == "__main__":
    main()

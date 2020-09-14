import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler

import dislib as ds
from dislib.cluster import KMeans, DBSCAN, GaussianMixture


def main():
    np.random.seed(0)

    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    n_samples = 1500
    noisy_circles = make_circles(n_samples=n_samples, factor=.5, noise=.05,
                                 random_state=170)
    noisy_moons = make_moons(n_samples=n_samples, noise=.05)
    blobs = make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5],
                        random_state=random_state)

    # ============
    # Set up cluster parameters
    # ============
    plt.figure(figsize=(9 * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    plot_num = 1

    default_base = {'quantile': .3, 'eps': .3, 'damping': .9,
                    'preference': -200, 'n_neighbors': 10, 'n_clusters': 3}

    datasets = [(noisy_circles,
                 {'damping': .77, 'preference': -240, 'quantile': .2,
                  'n_clusters': 2}), (noisy_moons,
                                      {'damping': .75, 'preference': -220,
                                       'n_clusters': 2}),
                (varied, {'eps': .18, 'n_neighbors': 2}),
                (aniso, {'eps': .15, 'n_neighbors': 2}), (blobs, {}),
                (no_structure, {})]

    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # ============
        # Create cluster objects
        # ============
        kmeans = KMeans(n_clusters=params["n_clusters"])
        dbscan = DBSCAN(eps=params["eps"], n_regions=1)
        gm = GaussianMixture(n_components=params["n_clusters"])

        clustering_algorithms = (('K-Means', kmeans), ('DBSCAN', dbscan),
                                 ('Gaussian mixture', gm))

        for name, algorithm in clustering_algorithms:
            t0 = time.time()

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        message="the number of connected "
                                                "components of the "
                                                "connectivity matrix is ["
                                                "0-9]{1,2} > 1. Completing "
                                                "it to avoid stopping the "
                                                "tree early.",
                                        category=UserWarning)
                warnings.filterwarnings("ignore", message="Graph is not fully "
                                                          "connected, "
                                                          "spectral "
                                                          "embedding may not "
                                                          "work as "
                                                          "expected.",
                                        category=UserWarning)

                data = ds.array(X, block_size=(300, 2))
                algorithm.fit(data)

            t1 = time.time()
            y_pred = algorithm.fit_predict(data).collect()

            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)

            colors = np.array(list(islice(cycle(
                ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
                 '#984ea3', '#999999', '#e41a1c', '#dede00']),
                int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1

    plt.show()


if __name__ == "__main__":
    main()

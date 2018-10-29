import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from dislib.data import Dataset
from dislib.ml.clustering import KMeans


def main():
    """
    Usage example copied from sciki-learn's webpage.

    """
    plt.figure(figsize=(12, 12))

    n_samples = 1500
    random_state = 170
    x, y = make_blobs(n_samples=n_samples, random_state=170)

    data = gen_data(x)

    # Incorrect number of clusters
    kmeans = KMeans(n_clusters=2, random_state=random_state, max_iter=300)
    kmeans.fit(data)
    y_pred = kmeans.predict(x)

    plt.subplot(221)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    centers = kmeans.centers
    plt.scatter(centers[:, 0], centers[:, 1], c="red")
    plt.title("Incorrect Number of Blobs")

    # Anisotropicly distributed data
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    x_aniso = np.dot(x, transformation)

    data = gen_data(x_aniso)

    kmeans = KMeans(n_clusters=3, random_state=random_state, max_iter=300)
    kmeans.fit(data)
    y_pred = kmeans.predict(x_aniso)

    plt.subplot(222)
    plt.scatter(x_aniso[:, 0], x_aniso[:, 1], c=y_pred)
    centers = kmeans.centers
    plt.scatter(centers[:, 0], centers[:, 1], c="red")
    plt.title("Anisotropicly Distributed Blobs")

    # Different variance
    x_varied, y_varied = make_blobs(n_samples=n_samples,
                                    cluster_std=[1.0, 2.5, 0.5],
                                    random_state=random_state)

    data = gen_data(x_varied)

    kmeans = KMeans(n_clusters=3, random_state=random_state, max_iter=300)
    kmeans.fit(data)
    y_pred = kmeans.predict(x_varied)

    plt.subplot(223)
    plt.scatter(x_varied[:, 0], x_varied[:, 1], c=y_pred)
    centers = kmeans.centers
    plt.scatter(centers[:, 0], centers[:, 1], c="red")
    plt.title("Unequal Variance")

    # Unevenly sized blobs
    x_filtered = np.vstack((x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))

    data = gen_data(x_filtered)

    kmeans = KMeans(n_clusters=3, random_state=random_state, max_iter=300)
    kmeans.fit(data)
    y_pred = kmeans.predict(x_filtered)

    plt.subplot(224)
    plt.scatter(x_filtered[:, 0], x_filtered[:, 1], c=y_pred)
    centers = kmeans.centers
    plt.scatter(centers[:, 0], centers[:, 1], c="red")
    plt.title("Unevenly Sized Blobs")

    plt.show()


def gen_data(x):
    data = []
    for i in range(0, 1500, 250):
        data.append(Dataset(x[i: i + 250]))
    return data


if __name__ == "__main__":
    main()

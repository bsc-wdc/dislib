import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from dislib.cluster import KMeans
from dislib.data import load_data


def main():
    """
    Usage example copied from sciki-learn's webpage.

    """
    plt.figure(figsize=(12, 12))

    n_samples = 1500
    random_state = 170
    x, y = make_blobs(n_samples=n_samples, random_state=random_state)

    dataset = load_data(x, subset_size=300)

    # Incorrect number of clusters
    kmeans = KMeans(n_clusters=2, random_state=random_state)
    kmeans.fit_predict(dataset)
    y_pred = dataset.labels

    plt.subplot(221)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    centers = kmeans.centers
    plt.scatter(centers[:, 0], centers[:, 1], c="red")
    plt.title("Incorrect Number of Blobs")

    # Anisotropicly distributed data
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    x_aniso = np.dot(x, transformation)

    dataset = load_data(x_aniso, subset_size=300)

    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans.fit_predict(dataset)
    y_pred = dataset.labels

    plt.subplot(222)
    plt.scatter(x_aniso[:, 0], x_aniso[:, 1], c=y_pred)
    centers = kmeans.centers
    plt.scatter(centers[:, 0], centers[:, 1], c="red")
    plt.title("Anisotropicly Distributed Blobs")

    # Different variance
    x_varied, y_varied = make_blobs(n_samples=n_samples,
                                    cluster_std=[1.0, 2.5, 0.5],
                                    random_state=random_state)

    dataset = load_data(x_varied, subset_size=300)

    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans.fit_predict(dataset)
    y_pred = dataset.labels

    plt.subplot(223)
    plt.scatter(x_varied[:, 0], x_varied[:, 1], c=y_pred)
    centers = kmeans.centers
    plt.scatter(centers[:, 0], centers[:, 1], c="red")
    plt.title("Unequal Variance")

    # Unevenly sized blobs
    x_filtered = np.vstack((x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))

    dataset = load_data(x_filtered, subset_size=300)

    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans.fit_predict(dataset)
    y_pred = dataset.labels

    plt.subplot(224)
    plt.scatter(x_filtered[:, 0], x_filtered[:, 1], c=y_pred)
    centers = kmeans.centers
    plt.scatter(centers[:, 0], centers[:, 1], c="red")
    plt.title("Unevenly Sized Blobs")
    plt.show()


if __name__ == "__main__":
    main()

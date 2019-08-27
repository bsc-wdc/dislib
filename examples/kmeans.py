import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

import dislib as ds
from dislib.cluster import KMeans


def main():
    """
    Usage example copied from scikit-learn's webpage.

    """
    plt.figure(figsize=(12, 12))

    n_samples = 1500
    random_state = 170
    x, y = make_blobs(n_samples=n_samples, random_state=random_state)

    dis_x = ds.array(x, block_size=(300, 2))

    # Incorrect number of clusters
    kmeans = KMeans(n_clusters=2, random_state=random_state)
    y_pred = kmeans.fit_predict(dis_x).collect()

    plt.subplot(221)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    centers = kmeans.centers
    plt.scatter(centers[:, 0], centers[:, 1], c="red")
    plt.title("Incorrect Number of Blobs")

    # Anisotropicly distributed data
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    x_aniso = np.dot(x, transformation)

    dis_x_aniso = ds.array(x_aniso, block_size=(300, 2))

    kmeans = KMeans(n_clusters=3, random_state=random_state)
    y_pred = kmeans.fit_predict(dis_x_aniso).collect()

    plt.subplot(222)
    plt.scatter(x_aniso[:, 0], x_aniso[:, 1], c=y_pred)
    centers = kmeans.centers
    plt.scatter(centers[:, 0], centers[:, 1], c="red")
    plt.title("Anisotropicly Distributed Blobs")

    # Different variance
    x_varied, y_varied = make_blobs(n_samples=n_samples,
                                    cluster_std=[1.0, 2.5, 0.5],
                                    random_state=random_state)

    dis_x_varied = ds.array(x_varied, block_size=(300, 2))

    kmeans = KMeans(n_clusters=3, random_state=random_state)
    y_pred = kmeans.fit_predict(dis_x_varied).collect()

    plt.subplot(223)
    plt.scatter(x_varied[:, 0], x_varied[:, 1], c=y_pred)
    centers = kmeans.centers
    plt.scatter(centers[:, 0], centers[:, 1], c="red")
    plt.title("Unequal Variance")

    # Unevenly sized blobs
    x_filtered = np.vstack((x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))

    dis_x_filtered = ds.array(x_filtered, block_size=(300, 2))

    kmeans = KMeans(n_clusters=3, random_state=random_state)
    y_pred = kmeans.fit_predict(dis_x_filtered).collect()

    plt.subplot(224)
    plt.scatter(x_filtered[:, 0], x_filtered[:, 1], c=y_pred)
    centers = kmeans.centers
    plt.scatter(centers[:, 0], centers[:, 1], c="red")
    plt.title("Unevenly Sized Blobs")
    plt.show()


if __name__ == "__main__":
    main()

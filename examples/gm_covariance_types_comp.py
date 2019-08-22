import matplotlib.pyplot as plt
import numpy as np

import dislib as ds
from dislib.cluster import GaussianMixture


def main():
    # Based on tests.test_gm.GaussianMixtureTest.test_covariance_types
    # Copied code START
    """ Tests GaussianMixture covariance types """
    np.random.seed(0)
    n_samples = 600
    n_features = 2

    def create_anisotropic_dataset():
        """Create dataset with 2 anisotropic gaussians of different
        weight"""
        n0 = 2 * n_samples // 3
        n1 = n_samples // 3
        x0 = np.random.normal(size=(n0, n_features))
        x1 = np.random.normal(size=(n1, n_features))
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x0 = np.dot(x0, transformation)
        x1 = np.dot(x1, transformation) + [0, 3]
        x = np.concatenate((x0, x1))
        y = np.concatenate((np.zeros(n0), np.ones(n1)))
        return x, y

    def create_spherical_blobs_dataset():
        """Create dataset with 2 spherical gaussians of different weight,
        variance and position"""
        n0 = 2 * n_samples // 3
        n1 = n_samples // 3
        x0 = np.random.normal(size=(n0, 2), scale=0.5, loc=[2, 0])
        x1 = np.random.normal(size=(n1, 2), scale=2.5)
        x = np.concatenate((x0, x1))
        y = np.concatenate((np.zeros(n0), np.ones(n1)))
        return x, y

    def create_uncorrelated_dataset():
        """Create dataset with 2 gaussians forming a cross of uncorrelated
        variables"""
        n0 = 2 * n_samples // 3
        n1 = n_samples // 3
        x0 = np.random.normal(size=(n0, n_features))
        x1 = np.random.normal(size=(n1, n_features))
        x0 = np.dot(x0, [[1.2, 0], [0, 0.5]]) + [0, 3]
        x1 = np.dot(x1, [[0.4, 0], [0, 2.5]]) + [1, 0]
        x = np.concatenate((x0, x1))
        y = np.concatenate((np.zeros(n0), np.ones(n1)))
        return x, y

    def create_correlated_dataset():
        """Create dataset with 2 gaussians forming a cross of correlated
        variables"""
        x, y = create_uncorrelated_dataset()
        x = np.dot(x, [[1, 1], [-1, 1]])
        return x, y

    datasets = {'aniso': create_anisotropic_dataset(),
                'blobs': create_spherical_blobs_dataset(),
                'uncorr': create_uncorrelated_dataset(),
                'corr': create_correlated_dataset()}
    real_labels = {k: v[1] for k, v in datasets.items()}
    for k, v in datasets.items():
        datasets[k] = ds.array(v[0], block_size=(200, v[0].shape[1]))

    covariance_types = 'full', 'tied', 'diag', 'spherical'

    def compute_accuracy(real, predicted):
        """ Computes classification accuracy for binary (0/1) labels"""
        equal_labels = np.count_nonzero(predicted == real)
        equal_ratio = equal_labels / len(real)
        return max(equal_ratio, 1 - equal_ratio)

    pred_labels = {}
    for cov_type in covariance_types:
        pred_labels[cov_type] = {}
        gm = GaussianMixture(n_components=2, covariance_type=cov_type,
                             random_state=0)
        for k, x in datasets.items():
            pred_labels[cov_type][k] = gm.fit_predict(x)
    accuracy = {}
    for cov_type in covariance_types:
        accuracy[cov_type] = {}
        for k, pred in pred_labels[cov_type].items():
            pred = pred.collect()
            pred_labels[cov_type][k] = pred
            accuracy[cov_type][k] = compute_accuracy(real_labels[k], pred)
    # Copied code END

    # Plot START
    plt.figure(figsize=(9 * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96,
                        wspace=.05,
                        hspace=.01)
    plot_num = 1

    for i_ds, (ds_name, x) in enumerate(datasets.items()):
        x = x.collect()

        plt.subplot(len(datasets), len(covariance_types) + 1, plot_num)
        if i_ds == 0:
            plt.title('original', size=18)
        colors = np.array(['#377eb8', '#ff7f00'])
        label_colors = colors[real_labels[ds_name].astype(int)]
        plt.scatter(x[:, 0], x[:, 1], s=10, color=label_colors)
        plt.xticks(())
        plt.yticks(())
        plot_num += 1

        for cov_type in covariance_types:
            plt.subplot(len(datasets), len(covariance_types) + 1, plot_num)
            if i_ds == 0:
                plt.title(cov_type, size=18)

            colors = np.array(['#377eb8', '#ff7f00'])
            label_colors = colors[pred_labels[cov_type][ds_name]]
            plt.scatter(x[:, 0], x[:, 1], s=10, color=label_colors)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01,
                     ('%.2f' % accuracy[cov_type][ds_name]).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1

    plt.show()
    # Plot END


if __name__ == "__main__":
    main()

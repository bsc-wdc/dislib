import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from pycompss.api.api import compss_wait_on
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import dislib as ds
from dislib.classification import CascadeSVM, RandomForestClassifier


def main():
    h = .02  # step size in the mesh

    names = ["Linear C-SVM", "RBF C-SVM", "Random forest"]

    classifiers = [
        CascadeSVM(kernel="linear", c=0.025, max_iter=5),
        CascadeSVM(gamma=2, c=1, max_iter=5),
        RandomForestClassifier(random_state=1)
    ]

    x, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    x += 2 * rng.uniform(size=x.shape)
    linearly_separable = (x, y)

    datasets = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable
                ]

    preprocessed_data = dict()
    scores = dict()
    mesh_accuracy_ds = dict()
    for ds_cnt, data in enumerate(datasets):
        # preprocess dataset, split into training and test part
        x, y = data
        x = StandardScaler().fit_transform(x)
        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=.4, random_state=42)
        ds_x_train = ds.array(x_train, block_size=(20, 2))
        ds_y_train = ds.array(y_train.reshape(-1, 1), block_size=(20, 1))
        ds_x_test = ds.array(x_test, block_size=(20, 2))
        ds_y_test = ds.array(y_test.reshape(-1, 1), block_size=(20, 1))

        x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
        y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        preprocessed_data[ds_cnt] = x, x_train, x_test, y_train, y_test, xx, yy

        for name, clf in zip(names, classifiers):
            clf.fit(ds_x_train, ds_y_train)
            scores[(ds_cnt, name)] = clf.score(ds_x_test, ds_y_test)

            mesh = np.c_[xx.ravel(), yy.ravel()]
            mesh_array = ds.array(mesh, (mesh.shape[0], 2))

            if hasattr(clf, "decision_function"):
                mesh_proba = clf.decision_function(mesh_array)
            else:
                mesh_proba = clf.predict_proba(mesh_array)
            mesh_accuracy_ds[(ds_cnt, name)] = mesh_proba

    # Synchronize while plotting the results
    plt.figure(figsize=(27, 9))
    i = 1
    for ds_cnt, data in enumerate(datasets):
        x, x_train, x_test, y_train, y_test, xx, yy = preprocessed_data[ds_cnt]

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

            score = compss_wait_on(scores[(ds_cnt, name)])
            mesh_proba = mesh_accuracy_ds[(ds_cnt, name)]

            if hasattr(clf, "decision_function"):
                Z = mesh_proba.collect()
            else:
                Z = mesh_proba.collect()[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot the training points
            ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='k')
            # Plot the testing points
            ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright,
                       edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

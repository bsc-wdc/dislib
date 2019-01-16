from sklearn.datasets import load_iris

from dislib.classification import RandomForestClassifier
from dislib.data import load_data

import numpy as np


def main():
    x, y = load_iris(return_X_y=True)

    iris_ds = load_data(x[::2], 10, y[::2])
    forest = RandomForestClassifier(10)
    forest.fit(iris_ds)

    test_ds = load_data(x[1::2], 10, y[1::2])
    forest.predict(test_ds)
    test_ds.collect()

    labels = []
    for subset in test_ds:
        labels.append(subset.labels)
    y_pred = np.concatenate(labels)

    print('y:')
    print(y[1::2])
    print('y_pred:')
    print(y_pred)
    test_ds = load_data(x[1::2], 10, y[1::2])
    print(forest.score(test_ds))


if __name__ == "__main__":
    main()

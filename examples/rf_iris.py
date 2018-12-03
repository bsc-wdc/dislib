from sklearn.datasets import load_iris

from dislib.classification import RandomForestClassifier
from dislib.data import Dataset, Subset, load_data

import numpy as np


def main():
    x, y = load_iris(return_X_y=True)

    iris_ds = load_data(x, len(x)//2, y)
    forest = RandomForestClassifier(10)
    forest.fit(iris_ds)

    test_ds = load_data(x, len(x)//2)
    prediction = forest.predict(test_ds)
    prediction.collect()

    labels = []
    for subset in prediction:
        labels.append(subset.labels)
    y_pred = np.concatenate(labels)

    print('y:')
    print(y)
    print('y_pred:')
    print(y_pred)


if __name__ == "__main__":
    main()

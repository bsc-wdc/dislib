from random import shuffle

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from dislib.classification import RandomForestClassifier
from dislib.data import load_data


def main():
    x, y = load_iris(return_X_y=True)

    indices = np.arange(len(x))
    shuffle(indices)

    # use 80% of samples for training
    train_idx = indices[:int(0.8 * len(x))]
    test_idx = indices[int(0.8 * len(x)):]

    # Train the RF classifier
    print("- Training Random Forest classifier with %s samples of Iris "
          "dataset." % len(train_idx))
    train_ds = load_data(x[train_idx], 10, y[train_idx])
    forest = RandomForestClassifier(10)
    forest.fit(train_ds)

    # Test the trained RF classifier
    print("- Testing the classifier.", end='')
    test_ds = load_data(x[test_idx], 10, y[test_idx])
    forest.predict(test_ds)
    test_ds.collect()

    labels = []
    for subset in test_ds:
        labels.append(subset.labels)
    y_pred = np.concatenate(labels)

    # Put results in fancy dataframe and print the accuracy
    df = pd.DataFrame(data=list(zip(y[test_idx], y_pred)),
                      columns=['Label', 'Predicted'])
    print(" Predicted values: \n\n%s" % df)
    print("test_ds %s" % test_ds)
    print("\n- Classifier accuracy: %s" % forest.score(test_ds))


if __name__ == "__main__":
    main()

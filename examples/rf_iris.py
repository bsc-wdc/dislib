from random import shuffle

import numpy as np
import pandas as pd
from pycompss.api.api import compss_barrier
from sklearn.datasets import load_iris

from dislib import array
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
    x_train = array(x[train_idx], (10, 4))
    y_train = array(y[train_idx][:, np.newaxis], (10, 1))
    forest = RandomForestClassifier(10)
    forest.fit(x_train, y_train)

    # Test the trained RF classifier
    print("- Testing the classifier.", end='')
    x_test = array(x[test_idx], (10, 4))
    y_real = array(y[test_idx][:, np.newaxis], (10, 1))
    y_pred = forest.predict(x_test)

    score = forest.score(x_test, y_real)

    # Put results in fancy dataframe and print the accuracy
    df = pd.DataFrame(data=list(zip(y[test_idx], y_pred.collect())),
                      columns=['Label', 'Predicted'])
    print(" Predicted values: \n\n%s" % df)
    print("\n- Classifier accuracy: %s" % score)


if __name__ == "__main__":
    main()

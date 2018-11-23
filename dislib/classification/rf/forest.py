import os
import warnings
from collections import Counter
from math import sqrt

import numpy as np
from numpy.lib import format
from pycompss.api.api import compss_wait_on

from dislib.classification.rf.decision_tree import DecisionTreeClassifier
from dislib.classification.rf.decision_tree import get_features_file
from dislib.classification.rf.decision_tree import get_y


class RandomForestClassifier:
    def __init__(self,
                 path_in,
                 n_instances,
                 n_features,
                 path_out,
                 n_estimators=10,
                 max_depth=None,
                 distr_depth=None,
                 try_features=None):
        self.path_in = path_in
        self.n_instances = n_instances
        self.n_features = n_features
        self.path_out = path_out
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.distr_depth = distr_depth
        if try_features is None:
            self.try_features = max(1, int(sqrt(n_features)))
        elif try_features == 'sqrt':
            self.try_features = max(1, int(sqrt(n_features)))
        elif try_features == 'third':
            self.try_features = max(1, int(n_features / 3))
        else:
            self.try_features = int(try_features)

        self.y = None
        self.y_codes = None
        self.n_classes = None
        self.trees = []

    def fit(self):
        """
        Fits the RandomForestClassifier.
        """
        features_file = get_features_file(self.path_in)
        if features_file is not None:
            self._features_file_check(features_file)
        self.y, self.y_codes, self.n_classes = get_y(self.path_in)

        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(self.path_in, self.n_instances,
                                          self.n_features, self.path_out,
                                          'tree_' + str(i), self.max_depth,
                                          self.distr_depth, True,
                                          self.try_features)
            tree.y_codes = self.y_codes
            tree.n_classes = self.n_classes
            self.trees.append(tree)

        for tree in self.trees:
            tree.fit()

        self.y, self.y_codes, self.n_classes = compss_wait_on(self.y,
                                                              self.y_codes,
                                                              self.n_classes)

        for tree in self.trees:
            tree.n_classes = self.n_classes

    def predict_probabilities(self, x_test):
        """ Predicts class probabilities by class code using a fitted forest
        and returns a 1D or 2D array. """

        return np.sum(tree.predict_probabilities(x_test) for tree
                      in self.trees) / len(self.trees)

    def predict(self, file_name='x_test.npy', soft_voting=True):
        """ Predicts classes using a fitted forest and returns an integer
        or an array. """
        try:
            x_test = np.load(os.path.join(self.path_in, file_name),
                             allow_pickle=False)
        except IOError:
            warnings.warn(
                'The test data file does not exist or cannot be read.')
            return

        if soft_voting:
            probabilities = self.predict_probabilities(x_test)
            return self.y.categories[np.argmax(probabilities, axis=1)]

        if len(x_test.shape) == 1:
            predicted = Counter(tree.predict(x_test) for tree in self.trees) \
                .most_common(1)[0][0]
            return self.y.categories[predicted]  # Convert code to real value
        elif len(x_test.shape) == 2:
            my_array = np.empty((len(self.trees), len(x_test)), np.int64)
            for i, tree in enumerate(self.trees):
                my_array[i, :] = tree.predict(x_test)
            predicted = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x)),
                0, my_array)
            return self.y.categories[predicted]  # Convert codes to real values
        else:
            raise ValueError

    def _features_file_check(self, features_file):
        with open(features_file, 'rb') as fp:
            version = format.read_magic(fp)
            try:
                format._check_version(version)
            except ValueError:
                raise ValueError('Unknown version of the features file.')
            shape, fortran_order, dtype = format._read_array_header(fp,
                                                                    version)
            if len(shape) != 2:
                raise ValueError(
                    'Cannot read 2D array from the features file.')
            if (self.n_features, self.n_instances) != shape:
                raise ValueError('The dimensions of the features file are '
                                 'different than the given dimensions.')
            if fortran_order:
                raise ValueError(
                    'Fortran order unsupported for features array')
            if dtype != np.float32:
                warnings.warn(
                    'Datatype ' + str(dtype) + ' has not been tested')

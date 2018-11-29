import math

from pycompss.api.task import task

from dislib.classification.rf.decision_tree import DecisionTreeClassifier

import numpy as np

from .data import RfDataset, transform_to_rf_dataset
from dislib.data import Dataset


class RandomForestClassifier:
    def __init__(self,
                 n_estimators=10,
                 try_features='sqrt',
                 max_depth=np.inf,
                 distr_depth='auto'):
        self.n_estimators = n_estimators
        self.try_features = try_features
        self.max_depth = max_depth
        self.distr_depth = distr_depth

        self.classes = None
        self.trees = []

    def fit(self, dataset):
        """
        Fits the RandomForestClassifier.

        :param dataset: dislib.data.Dataset.
        """

        if not isinstance(dataset, (Dataset, RfDataset)):
            raise TypeError('Invalid type for param dataset.')
        if isinstance(dataset, Dataset):
            dataset = transform_to_rf_dataset(dataset)

        if isinstance(dataset.features_path, str):
            dataset.validate_features_file()

        self.try_features = resolve_try_features(self.try_features, dataset.get_n_features())
        self.distr_depth = resolve_distr_depth(self.distr_depth, self.max_depth, dataset.get_n_features())
        self.classes = dataset.get_classes()

        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(self.try_features, self.max_depth, self.distr_depth, bootstrap=True)
            self.trees.append(tree)

        for tree in self.trees:
            tree.fit(dataset)

    def predict_proba(self, dataset):
        """ Predicts class probabilities using a fitted forest. The order of the classes is given by self.classes. """
        assert self.trees is not None, 'The random forest is not fitted.'
        for subset in dataset:
            tree_predictions = []
            for tree in self.trees:
                tree_predictions.append(tree.predict_proba(subset.samples))
            subset.labels = join_predictions(*tree_predictions)
        return dataset  # TODO: required?

    def predict(self, dataset, soft_voting=True):
        """ Predicts classes using a fitted forest. """
        assert self.trees is not None, 'The random forest is not fitted.'
        if soft_voting:
            for subset in dataset:
                tree_predictions = []
                for tree in self.trees:
                    tree_predictions.append(tree.predict_proba(subset.samples))
                subset.labels = soft_vote(self.classes, *tree_predictions)
        else:
            for subset in dataset:
                tree_predictions = []
                for tree in self.trees:
                    tree_predictions.append(tree.predict(subset.samples))
                subset.labels = hard_vote(self.classes, *tree_predictions)


def resolve_try_features(try_features, n_features):
    if try_features is None:
        return n_features
    elif try_features == 'sqrt':
        return int(math.sqrt(n_features))
    elif try_features == 'third':
        return max(1, n_features // 3)
    else:
        return int(try_features)


def resolve_distr_depth(distr_depth, max_depth,  n_samples):
    if distr_depth is 'auto':
        distr_depth = max(0, int(math.log10(n_samples)) - 4)
        distr_depth = min(distr_depth, max_depth)
    return distr_depth


@task(returns=1)
def join_predictions(*predictions):
    return sum(*predictions)/len(predictions)


@task(returns=1)
def soft_vote(classes, *predictions):
    return classes[np.argmax(sum(*predictions), axis=1)]


@task(returns=1)
def hard_vote(classes, *predictions):
    return classes[np.argmax(*predictions, axis=1)]

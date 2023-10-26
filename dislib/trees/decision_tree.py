from dislib.trees.mmap import (DecisionTreeClassifier as
                               DecisionTreeClassifierMMap)
from dislib.trees.mmap import (DecisionTreeRegressor as
                               DecisionTreeRegressorMMap)
from dislib.trees.distributed import (DecisionTreeClassifier as
                                      DecisionTreeClassifierDistributed)
from dislib.trees.distributed import (DecisionTreeRegressor as
                                      DecisionTreeRegressorDistributed)
from sklearn.tree import DecisionTreeClassifier as SklearnDTClassifier
from sklearn.tree import DecisionTreeRegressor as SklearnDTRegressor
from dislib.trees.distributed.decision_tree import (_RegressionNode,
                                                    _ClassificationNode)


class BaseDecisionTree:
    """Base class for distributed decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(
        self,
        try_features,
        max_depth,
        distr_depth,
        sklearn_max,
        bootstrap,
        random_state,
        base_node,
        base_tree,
        n_classes=None,
        range_min=None,
        range_max=None,
        n_split_points="auto",
        split_computation="raw",
        sync_after_fit=True,
        mmap=True,
    ):
        self.try_features = try_features
        self.max_depth = max_depth
        self.distr_depth = distr_depth
        self.sklearn_max = sklearn_max
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.base_node = base_node
        self.base_tree = base_tree

        self.n_features = None
        self.n_classes = None

        self.tree = None
        self.nodes_info = None
        self.subtrees = None

    def fit(self, dataset):
        """Fits the DecisionTree.

        Parameters
        ----------
        dataset : dislib.classification.rf._data.RfDataset
        """
        if self.mmap:
            if SklearnDTRegressor == self.base_tree:
                tree = DecisionTreeRegressorMMap
            else:
                tree = DecisionTreeClassifierMMap
            self.tree = tree(

            )
        else:
            if SklearnDTRegressor == self.base_tree:
                tree = DecisionTreeRegressorDistributed
            else:
                tree = DecisionTreeClassifierDistributed
            self.tree = tree(

            )

    def predict(self, x_row):
        """Predicts target values or classes for the given samples using
        a fitted tree.

        Parameters
        ----------
        x_row : ds-array
            A row block of samples.

        Returns
        -------
        predicted : ndarray
            An array with the predicted classes or values for the given
            samples. For classification, the values are codes of the fitted
            dislib.classification.rf.data.RfDataset. The returned object can
            be a pycompss.runtime.Future object.
        """

        assert self.tree is not None, "The decision tree is not fitted."
        if self.mmap:
            return self.tree.predict(x_row)
        else:
            return self.tree.predict(x_row, collect=False)


class DecisionTreeClassifier(BaseDecisionTree):
    """A distributed decision tree classifier.

    Parameters
    ----------
    try_features : int
        The number of features to consider when looking for the best split.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires
        to effectively inspect more than ``try_features`` features.
    max_depth : int
        The maximum depth of the tree. If np.inf, then nodes are expanded
        until all leaves are pure.
    distr_depth : int
        Number of levels of the tree in which the nodes are split in a
        distributed way.
    bootstrap : bool
        Randomly select n_instances samples with repetition (used in random
        forests).
    random_state : RandomState instance
        The random number generator.

    Attributes
    ----------
    n_features : int
        The number of features of the dataset. It can be a
        pycompss.runtime.Future object.
    n_classes : int
        The number of classes of this RfDataset. It can be a
        pycompss.runtime.Future object.
    tree : None or _Node
        The root node of the tree after the tree is fitted.
    nodes_info : None or list of _InnerNodeInfo and _LeafInfo
        List of the node information for the nodes of the tree in the same
        order as obtained in the fit() method, up to ``distr_depth`` depth.
        After fit(), it is a pycompss.runtime.Future object.
    subtrees : None or list of _Node
        List of subtrees of the tree at ``distr_depth`` depth  obtained in the
        fit() method. After fit(), it is a list of pycompss.runtime.Future
        objects.

    Methods
    -------
    fit(dataset)
        Fits the DecisionTreeClassifier.
    predict(x_row)
        Predicts classes for the given samples using a fitted tree.
    predict_proba(x_row)
        Predicts class probabilities for the given smaples using a fitted tree.

    """

    def __init__(
        self,
        try_features,
        max_depth,
        distr_depth,
        sklearn_max,
        bootstrap,
        random_state,
    ):
        super().__init__(
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            _ClassificationNode,
            SklearnDTClassifier,
        )

    def predict_proba(self, x_row):
        """Predicts class probabilities for a row block using a fitted tree.

        Parameters
        ----------
        x_row : ds-array
            A row block of samples.

        Returns
        -------
        predicted_proba : ndarray
            An array with the predicted probabilities for the given samples.
            The shape is (len(subset.samples), self.n_classes), with the index
            of the column being codes of the fitted
            dislib.classification.rf.data.RfDataset. The returned object can be
            a pycompss.runtime.Future object.
        """

        assert self.tree is not None, "The decision tree is not fitted."
        if self.mmap:
            return self.tree.predict_proba(x_row)
        else:
            return self.tree.predict_proba(x_row, collect=False)


class DecisionTreeRegressor(BaseDecisionTree):
    """A distributed decision tree regressor.

    Parameters
    ----------
    try_features : int
        The number of features to consider when looking for the best split.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires
        to effectively inspect more than ``try_features`` features.
    max_depth : int
        The maximum depth of the tree. If np.inf, then nodes are expanded
        until all leaves are pure.
    distr_depth : int
        Number of levels of the tree in which the nodes are split in a
        distributed way.
    bootstrap : bool
        Randomly select n_instances samples with repetition (used in random
        forests).
    random_state : RandomState instance
        The random number generator.

    Attributes
    ----------
    n_features : int
        The number of features of the dataset. It can be a
        pycompss.runtime.Future object.
    tree : None or _Node
        The root node of the tree after the tree is fitted.
    nodes_info : None or list of _InnerNodeInfo and _LeafInfo
        List of the node information for the nodes of the tree in the same
        order as obtained in the fit() method, up to ``distr_depth`` depth.
        After fit(), it is a pycompss.runtime.Future object.
    subtrees : None or list of _Node
        List of subtrees of the tree at ``distr_depth`` depth  obtained in the
        fit() method. After fit(), it is a list of pycompss.runtime.Future
        objects.

    Methods
    -------
    fit(dataset)
        Fits the DecisionTreeRegressor.
    predict(x_row)
        Predicts target values for the given samples using a fitted tree.
    """

    def __init__(
        self,
        try_features,
        max_depth,
        distr_depth,
        sklearn_max,
        bootstrap,
        random_state,
    ):
        super().__init__(
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            bootstrap,
            random_state,
            _RegressionNode,
            SklearnDTRegressor,
        )

import numpy as np
from sklearn.base import BaseEstimator

from dislib.trees.mmap import (DecisionTreeClassifier as
                               DecisionTreeClassifierMMap,
                               DecisionTreeRegressor as
                               DecisionTreeRegressorMMap)
from dislib.trees.mmap import (RandomForestClassifier as
                               RandomForestClassifierMMap,
                               RfClassifierDataset, RfRegressorDataset,
                               RandomForestRegressor as
                               RandomForestRegressorMMap)
from dislib.trees.distributed import (DecisionTreeClassifier as
                                      DecisionTreeClassifierDistributed,
                                      DecisionTreeRegressor as
                                      DecisionTreeRegressorDistributed)
from dislib.trees.distributed import (RandomForestClassifier as
                                      RandomForestClassifierDistributed,
                                      RandomForestRegressor as
                                      RandomForestRegressorDistributed)
from dislib.trees.nested import (DecisionTreeClassifier as
                                 DecisionTreeClassifierNested,
                                 DecisionTreeRegressor as
                                 DecisionTreeRegressorNested)
from dislib.trees.nested import (RandomForestClassifier as
                                 RandomForestClassifierNested,
                                 RandomForestRegressor as
                                 RandomForestRegressorNested)


class BaseRandomForest(BaseEstimator):
    """Base class for distributed random forests.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(
        self,
        n_estimators,
        try_features,
        max_depth,
        distr_depth,
        sklearn_max,
        hard_vote,
        random_state,
        base_tree,
        base_dataset=None,
        n_classes=None,
        range_max=None,
        range_min=None,
        bootstrap=True,
        n_split_points="auto",
        split_computation="raw",
        sync_after_fit=True,
        mmap=True,
        nested=False,
    ):
        self.n_estimators = n_estimators
        self.try_features = try_features
        self.max_depth = max_depth
        self.distr_depth = distr_depth
        self.sklearn_max = sklearn_max
        self.hard_vote = hard_vote
        self.random_state = random_state
        self.base_tree = base_tree
        self.base_dataset = base_dataset
        self.n_classes = n_classes
        self.range_max = range_max
        self.range_min = range_min
        self.bootstrap = bootstrap
        self.n_split_points = n_split_points
        self.split_computation = split_computation
        self.sync_after_fit = sync_after_fit
        self.mmap = mmap
        self.nested = nested
        self.rf = None

    def fit(self, x, y):
        """Fits a RandomForest.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``.
        y : ds-array, shape=(n_samples, 1)
            The target values.

        Returns
        -------
        self : RandomForest
        """
        if self.mmap:
            if DecisionTreeRegressorMMap == self.base_tree:
                self.rf = RandomForestRegressorMMap(
                    self.n_estimators, self.try_features, self.max_depth,
                    self.distr_depth, self.sklearn_max,
                    self.random_state)
            else:
                self.rf = RandomForestClassifierMMap(
                    self.n_estimators, self.try_features, self.max_depth,
                    self.distr_depth, self.sklearn_max,
                    self.hard_vote, self.random_state)
        else:
            if self.nested:
                if DecisionTreeRegressorNested == self.base_tree:
                    self.rf = RandomForestRegressorNested(
                        self.n_estimators, self.try_features,
                        self.max_depth, self.distr_depth,
                        self.sklearn_max, self.random_state,
                        range_max=self.range_max,
                        range_min=self.range_min,
                        bootstrap=self.bootstrap,
                        n_split_points=self.n_split_points,
                        split_computation=self.split_computation,
                        sync_after_fit=self.sync_after_fit)
                else:
                    self.rf = RandomForestClassifierNested(
                        self.n_classes, self.n_estimators,
                        self.try_features, self.max_depth,
                        self.distr_depth, self.sklearn_max,
                        self.hard_vote, self.random_state,
                        range_max=self.range_max, range_min=self.range_min,
                        bootstrap=self.bootstrap,
                        n_split_points=self.n_split_points,
                        split_computation=self.split_computation,
                        sync_after_fit=self.sync_after_fit)
            else:
                if DecisionTreeRegressorDistributed == self.base_tree:
                    self.rf = RandomForestRegressorDistributed(
                        self.n_estimators, self.try_features, self.max_depth,
                        self.distr_depth, self.sklearn_max,
                        self.random_state,
                        range_max=self.range_max, range_min=self.range_min,
                        bootstrap=self.bootstrap,
                        n_split_points=self.n_split_points,
                        split_computation=self.split_computation,
                        sync_after_fit=self.sync_after_fit)
                else:
                    self.rf = RandomForestClassifierDistributed(
                        self.n_classes, self.n_estimators,
                        self.try_features, self.max_depth,
                        self.distr_depth, self.sklearn_max,
                        self.hard_vote, self.random_state,
                        range_max=self.range_max, range_min=self.range_min,
                        bootstrap=self.bootstrap,
                        n_split_points=self.n_split_points,
                        split_computation=self.split_computation,
                        sync_after_fit=self.sync_after_fit)

        self.rf.fit(x, y)
        if self.mmap and DecisionTreeClassifierMMap == self.base_tree:
            self.classes = self.rf.classes
        return self

    def save_model(self, filepath, overwrite=True, save_format="json"):
        """Saves a model to a file.
        The model is synchronized before saving and can be reinstantiated in
        the exact same state, without any of the code used for model
        definition or fitting.
        Parameters
        ----------
        filepath : str
            Path where to save the model
        overwrite : bool, optional (default=True)
            Whether any existing model at the target
            location should be overwritten.
        save_format : str, optional (default='json)
            Format used to save the models.
        Examples
        --------
        >>> from dislib.trees import RandomForestClassifier
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> y = np.array([1, 1, 2, 2, 2, 1])
        >>> x_train = ds.array(x, (2, 2))
        >>> y_train = ds.array(y, (2, 1))
        >>> model = RandomForestClassifier(n_estimators=2, random_state=0)
        >>> model.fit(x_train, y_train)
        >>> model.save_model('/tmp/model')
        >>> loaded_model = RandomForestClassifier()
        >>> loaded_model.load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
        >>> model_pred = model.predict(x_test)
        >>> loaded_model_pred = loaded_model.predict(x_test)
        >>> assert np.allclose(model_pred.collect(),
        >>> loaded_model_pred.collect())
        """
        if self.rf is not None:
            self.rf.save_model(filepath,
                               overwrite=overwrite,
                               save_format=save_format)

    def load_model(self, filepath, load_format="json"):
        """Loads a model from a file.
        The model is reinstantiated in the exact same state in which it
        was saved, without any of the code used for model definition or
        fitting.
        Parameters
        ----------
        filepath : str
            Path of the saved the model
        load_format : str, optional (default='json')
            Format used to load the model.
        Examples
        --------
        >>> from dislib.trees import RandomForestClassifier
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> y = np.array([1, 1, 2, 2, 2, 1])
        >>> x_train = ds.array(x, (2, 2))
        >>> y_train = ds.array(y, (2, 1))
        >>> model = RandomForestClassifier(n_estimators=2, random_state=0)
        >>> model.fit(x_train, y_train)
        >>> model.save_model(model, '/tmp/model')
        >>> loaded_model = RandomForestClassifier()
        >>> loaded_model.load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
        >>> model_pred = model.predict(x_test)
        >>> loaded_model_pred = loaded_model.predict(x_test)
        >>> assert np.allclose(model_pred.collect(),
        """
        # Load model
        if self.rf is not None:
            self.rf.load_model(filepath,
                               load_format=load_format)
        else:
            if self.mmap:
                if DecisionTreeRegressorMMap == self.base_tree:
                    self.rf = RandomForestRegressorMMap(
                        self.n_estimators, self.try_features, self.max_depth,
                        self.distr_depth, self.sklearn_max,
                        self.random_state)
                else:
                    self.rf = RandomForestClassifierMMap(
                        self.n_estimators, self.try_features, self.max_depth,
                        self.distr_depth, self.sklearn_max,
                        self.hard_vote, self.random_state)
            else:
                if self.nested:
                    if DecisionTreeRegressorNested == self.base_tree:
                        self.rf = RandomForestRegressorNested(
                            self.n_estimators, self.try_features,
                            self.max_depth, self.distr_depth,
                            self.sklearn_max, self.random_state,
                            range_max=self.range_max,
                            range_min=self.range_min,
                            bootstrap=self.bootstrap,
                            n_split_points=self.n_split_points,
                            split_computation=self.split_computation,
                            sync_after_fit=self.sync_after_fit)
                    else:
                        self.rf = RandomForestClassifierNested(
                            self.n_classes, self.n_estimators,
                            self.try_features, self.max_depth,
                            self.distr_depth, self.sklearn_max,
                            self.hard_vote, self.random_state,
                            range_max=self.range_max, range_min=self.range_min,
                            bootstrap=self.bootstrap,
                            n_split_points=self.n_split_points,
                            split_computation=self.split_computation,
                            sync_after_fit=self.sync_after_fit)
                else:
                    if DecisionTreeRegressorDistributed == self.base_tree:
                        self.rf = RandomForestRegressorDistributed(
                            self.n_estimators, self.try_features,
                            self.max_depth, self.distr_depth,
                            self.sklearn_max, self.random_state,
                            range_max=self.range_max, range_min=self.range_min,
                            bootstrap=self.bootstrap,
                            n_split_points=self.n_split_points,
                            split_computation=self.split_computation,
                            sync_after_fit=self.sync_after_fit)
                    else:
                        self.rf = RandomForestClassifierDistributed(
                            self.n_classes, self.n_estimators,
                            self.try_features, self.max_depth,
                            self.distr_depth, self.sklearn_max,
                            self.hard_vote, self.random_state,
                            range_max=self.range_max, range_min=self.range_min,
                            bootstrap=self.bootstrap,
                            n_split_points=self.n_split_points,
                            split_computation=self.split_computation,
                            sync_after_fit=self.sync_after_fit)
            self.rf.load_model(filepath,
                               load_format=load_format)


class RandomForestClassifier(BaseRandomForest):
    """A distributed random forest classifier.

    Parameters
    ----------
    n_estimators : int, optional (default=10)
        Number of trees to fit.
    try_features : int, str or None, optional (default='sqrt')
        The number of features to consider when looking for the best split:

        - If "sqrt", then `try_features=sqrt(n_features)`.
        - If "third", then `try_features=n_features // 3`.
        - If None, then `try_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires
        to effectively inspect more than ``try_features`` features.
    max_depth : int or np.inf, optional (default=np.inf)
        The maximum depth of the tree. If np.inf, then nodes are expanded
        until all leaves are pure.
    distr_depth : int or str, optional (default='auto')
        Number of levels of the tree in which the nodes are split in a
        distributed way.
    sklearn_max: int or float, optional (default=1e8)
        Maximum size (len(subsample)*n_features) of the arrays passed to
        sklearn's DecisionTreeClassifier.fit(), which is called to fit subtrees
        (subsamples) of our DecisionTreeClassifier. sklearn fit() is used
        because it's faster, but requires loading the data to memory, which can
        cause memory problems for large datasets. This parameter can be
        adjusted to fit the hardware capabilities.
    hard_vote : bool, optional (default=False)
        If True, it uses majority voting over the predict() result of the
        decision tree predictions. If False, it takes the class with the higher
        probability given by predict_proba(), which is an average of the
        probabilities given by the decision trees.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    n_classes : int
        Number of classes that appear on the dataset. Only needed on
        distributed random forest.
    range_min : ds-array or np.array
        Contains the minimum values of the different attributes of the dataset
        Only used on distributed random forest (it is an optional parameter)
    range_max : ds-array or np.array
        Contains the maximum values of the different attributes of the dataset
        Only used on distributed random forest (it is an optional parameter)
    n_split_points : String or int
        Number of split points to evaluate.
        "auto", "sqrt" or integer value.
        Used on distributed random forest (non memory map version)
    split_computation : String
        "raw", "gaussian_approximation" or "uniform_approximation"
        distribution of the values followed by the split points selected.
        Used on distributed random forest (non memory map version)
    sync_after_fit : bool
        Synchronize or not after the training.
        Used on distributed random forest (non memory map version)
    mmap : bool
        Use the memory map version or not.
    nested : bool
        Use the nested version or not.

    Attributes
    ----------
    classes : None or ndarray
        Array of distinct classes, set at fit().
    rf : RandomForestClassifier selected
        Instance of mmap, distributed or nested
        RandomForestClassifier selected.
    """

    def __init__(
        self,
        n_estimators=10,
        try_features="sqrt",
        max_depth=np.inf,
        distr_depth="auto",
        sklearn_max=1e8,
        hard_vote=False,
        random_state=None,
        n_classes=None,
        range_max=None,
        range_min=None,
        bootstrap=True,
        n_split_points="auto",
        split_computation="raw",
        sync_after_fit=True,
        mmap=True,
        nested=False,
    ):
        if mmap:
            super().__init__(
                n_estimators,
                try_features,
                max_depth,
                distr_depth,
                sklearn_max,
                hard_vote,
                random_state,
                base_tree=DecisionTreeClassifierMMap,
                base_dataset=RfClassifierDataset,
            )
        else:
            if nested:
                super().__init__(
                    n_estimators,
                    try_features,
                    max_depth,
                    distr_depth,
                    sklearn_max,
                    hard_vote,
                    random_state,
                    base_tree=DecisionTreeClassifierNested,
                    base_dataset=None,
                    n_classes=n_classes,
                    range_max=range_max,
                    range_min=range_min,
                    bootstrap=bootstrap,
                    n_split_points=n_split_points,
                    split_computation=split_computation,
                    sync_after_fit=sync_after_fit,
                    mmap=mmap,
                    nested=nested,
                )
            else:
                super().__init__(
                    n_estimators,
                    try_features,
                    max_depth,
                    distr_depth,
                    sklearn_max,
                    hard_vote,
                    random_state,
                    base_tree=DecisionTreeClassifierDistributed,
                    base_dataset=None,
                    n_classes=n_classes,
                    range_max=range_max,
                    range_min=range_min,
                    bootstrap=bootstrap,
                    n_split_points=n_split_points,
                    split_computation=split_computation,
                    sync_after_fit=sync_after_fit,
                    mmap=mmap,
                    nested=nested
                )

    def predict(self, x):
        """Predicts target classes using a fitted forest.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ds-array, shape=(n_samples, 1)
            Predicted class labels for x.
        """
        assert self.rf is not None, "The random forest is not fitted."
        return self.rf.predict(x)

    def predict_proba(self, x):
        """Predicts class probabilities using a fitted forest.

        The probabilities are obtained as an average of the probabilities of
        each decision tree.


        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            The input samples.

        Returns
        -------
        probabilities : ds-array, shape=(n_samples, n_classes)
            Predicted probabilities for the samples to belong to each class.
            The columns of the array correspond to the classes given at
            self.classes.
        """
        assert self.rf is not None, "The random forest is not fitted."
        return self.rf.predict_proba(x)

    def score(self, x, y, collect=False):
        """Accuracy classification score.

        Returns the mean accuracy of the predictions on the given test data.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            The training input samples.
        y : ds-array, shape (n_samples, 1)
            The true labels.
        collect : bool, optional (default=False)
            When True, a synchronized result is returned.


        Returns
        -------
        score : float (as future object)
            Fraction of correctly classified samples.
        """
        assert self.rf is not None, "The random forest is not fitted."
        return self.rf.score(x, y, collect=collect)

    def load_model(self, filepath, load_format="json"):
        """Loads a model from a file.
        The model is reinstantiated in the exact same state in which it
        was saved, without any of the code used for model definition or
        fitting.
        Parameters
        ----------
        filepath : str
            Path of the saved the model
        load_format : str, optional (default='json')
            Format used to load the model.
        Examples
        --------
        >>> from dislib.trees import RandomForestClassifier
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> y = np.array([1, 1, 2, 2, 2, 1])
        >>> x_train = ds.array(x, (2, 2))
        >>> y_train = ds.array(y, (2, 1))
        >>> model = RandomForestClassifier(n_estimators=2, random_state=0)
        >>> model.fit(x_train, y_train)
        >>> model.save_model('/tmp/model')
        >>> loaded_model = RandomForestClassifier()
        >>> loaded_model.load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
        >>> model_pred = model.predict(x_test)
        >>> loaded_model_pred = loaded_model.predict(x_test)
        >>> assert np.allclose(model_pred.collect(),
        """
        super().load_model(filepath, load_format=load_format)

    def save_model(self, filepath, overwrite=True, save_format="json"):
        """Saves a model to a file.
        The model is synchronized before saving and can be reinstantiated in
        the exact same state, without any of the code used for model
        definition or fitting.
        Parameters
        ----------
        filepath : str
            Path where to save the model
        overwrite : bool, optional (default=True)
            Whether any existing model at the target
            location should be overwritten.
        save_format : str, optional (default='json)
            Format used to save the models.
        Examples
        --------
        >>> from dislib.trees import RandomForestClassifier
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> y = np.array([1, 1, 2, 2, 2, 1])
        >>> x_train = ds.array(x, (2, 2))
        >>> y_train = ds.array(y, (2, 1))
        >>> model = RandomForestClassifier(n_estimators=2, random_state=0)
        >>> model.fit(x_train, y_train)
        >>> model.save_model('/tmp/model')
        >>> loaded_model = RandomForestClassifier()
        >>> loaded_model.load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
        >>> model_pred = model.predict(x_test)
        >>> loaded_model_pred = loaded_model.predict(x_test)
        >>> assert np.allclose(model_pred.collect(),
        >>> loaded_model_pred.collect())
        """
        super().save_model(
            filepath,
            overwrite=overwrite,
            save_format=save_format
        )


class RandomForestRegressor(BaseRandomForest):
    """A distributed random forest regressor.

    Parameters
    ----------
    n_estimators : int, optional (default=10)
        Number of trees to fit.
    try_features : int, str or None, optional (default='sqrt')
        The number of features to consider when looking for the best split:

        - If "sqrt", then `try_features=sqrt(n_features)`.
        - If "third", then `try_features=n_features // 3`.
        - If None, then `try_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires
        to effectively inspect more than ``try_features`` features.
    max_depth : int or np.inf, optional (default=np.inf)
        The maximum depth of the tree. If np.inf, then nodes are expanded
        until all leaves are pure.
    distr_depth : int or str, optional (default='auto')
        Number of levels of the tree in which the nodes are split in a
        distributed way.
    sklearn_max: int or float, optional (default=1e8)
        Maximum size (len(subsample)*n_features) of the arrays passed to
        sklearn's DecisionTreeRegressor.fit(), which is called to fit subtrees
        (subsamples) of our DecisionTreeRegressor. sklearn fit() is used
        because it's faster, but requires loading the data to memory, which can
        cause memory problems for large datasets. This parameter can be
        adjusted to fit the hardware capabilities.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    n_classes : int
        Number of classes that appear on the dataset. Only needed on
        distributed random forest.
    range_min : ds-array or np.array
        Contains the minimum values of the different attributes of the dataset
        Only used on distributed random forest (it is an optional parameter)
    range_max : ds-array or np.array
        Contains the maximum values of the different attributes of the dataset
        Only used on distributed random forest (it is an optional parameter)
    n_split_points : String or int
        Number of split points to evaluate.
        "auto", "sqrt" or integer value.
        Used on distributed random forest (non memory map version)
    split_computation : String
        "raw", "gaussian_approximation" or "uniform_approximation"
        distribution of the values followed by the split points selected.
        Used on distributed random forest (non memory map version)
    sync_after_fit : bool
        Synchronize or not after the training.
        Used on distributed random forest (non memory map version)
    mmap : bool
        Use the memory map version of the algorithm or not
    nested : bool
        Use the nested version of the algorithm or not

    Attributes
    ----------
     rf : RandomForestRegressor selected
        Instance of mmap, distributed or nested
        RandomForestRegressor selected.
    """

    def __init__(
        self,
        n_estimators=10,
        try_features="sqrt",
        max_depth=np.inf,
        distr_depth="auto",
        sklearn_max=1e8,
        random_state=None,
        range_max=None,
        range_min=None,
        bootstrap=True,
        n_split_points="auto",
        split_computation="raw",
        sync_after_fit=True,
        mmap=True,
        nested=False,
    ):
        hard_vote = None
        if mmap:
            super().__init__(
                n_estimators,
                try_features,
                max_depth,
                distr_depth,
                sklearn_max,
                hard_vote,
                random_state,
                base_tree=DecisionTreeRegressorMMap,
                base_dataset=RfRegressorDataset,
            )
        else:
            if nested:
                super().__init__(
                    n_estimators,
                    try_features,
                    max_depth,
                    distr_depth,
                    sklearn_max,
                    hard_vote,
                    random_state,
                    base_tree=DecisionTreeRegressorNested,
                    base_dataset=None,
                    range_max=range_max,
                    range_min=range_min,
                    bootstrap=bootstrap,
                    n_split_points=n_split_points,
                    split_computation=split_computation,
                    sync_after_fit=sync_after_fit,
                    mmap=mmap,
                    nested=nested,
                )
            else:
                super().__init__(
                    n_estimators,
                    try_features,
                    max_depth,
                    distr_depth,
                    sklearn_max,
                    hard_vote,
                    random_state,
                    base_tree=DecisionTreeRegressorDistributed,
                    base_dataset=None,
                    range_max=range_max,
                    range_min=range_min,
                    bootstrap=bootstrap,
                    n_split_points=n_split_points,
                    split_computation=split_computation,
                    sync_after_fit=sync_after_fit,
                    mmap=mmap,
                    nested=nested,
                )

    def predict(self, x):
        """Predicts target values using a fitted forest.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ds-array, shape=(n_samples, 1)
            Predicted values for x.
        """
        assert self.rf is not None, "The random forest is not fitted."
        return self.rf.predict(x)

    def score(self, x, y, collect=False):
        """R2 regression score.

        Returns the coefficient of determination $R^2$ of the prediction.
        The coefficient $R^2$ is defined as $(1-u/v)$, where $u$
        is the residual sum of squares `((y_true - y_pred) ** 2).sum()` and
        $v$ is the total sum of squares
        `((y_true - y_true.mean()) ** 2).sum()`.
        The best possible score is 1.0 and it can be negative
        if the model is arbitrarily worse.
        A constant model that always predicts the expected value of y,
        disregarding the input features, would get a $R^2$ score of 0.0.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            The training input samples.
        y : ds-array, shape (n_samples, 1)
            The true values.
        collect : bool, optional (default=False)
            When True, a synchronized result is returned.


        Returns
        -------
        score : float (as future object)
            Coefficient of determination $R^2$.
        """
        assert self.rf is not None, "The random forest is not fitted."
        return self.rf.score(x, y, collect=collect)

    def load_model(self, filepath, load_format="json"):
        """Loads a model from a file.
        The model is reinstantiated in the exact same state in which it
        was saved, without any of the code used for model definition or
        fitting.
        Parameters
        ----------
        filepath : str
            Path of the saved the model
        load_format : str, optional (default='json')
            Format used to load the model.
        Examples
        --------
        >>> from dislib.trees import RandomForestRegressor
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> y = np.array([1.5, 1.2, 2.7, 2.1, 0.2, 0.6])
        >>> x_train = ds.array(x, (2, 2))
        >>> y_train = ds.array(y, (2, 1))
        >>> model = RandomForestRegressor(n_estimators=2, random_state=0)
        >>> model.fit(x_train, y_train)
        >>> model.save_model('/tmp/model')
        >>> loaded_model = RandomForestRegressor()
        >>> loaded_model.load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
        >>> model_pred = model.predict(x_test)
        >>> loaded_model_pred = loaded_model.predict(x_test)
        >>> assert np.allclose(model_pred.collect(),
        """
        super().load_model(filepath, load_format=load_format)

    def save_model(self, filepath, overwrite=True, save_format="json"):
        """Saves a model to a file.
        The model is synchronized before saving and can be reinstantiated in
        the exact same state, without any of the code used for model
        definition or fitting.
        Parameters
        ----------
        filepath : str
            Path where to save the model
        overwrite : bool, optional (default=True)
            Whether any existing model at the target
            location should be overwritten.
        save_format : str, optional (default='json)
            Format used to save the models.
        Examples
        --------
        >>> from dislib.trees import RandomForestRegressor
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> y = np.array([1.5, 1.2, 2.7, 2.1, 0.2, 0.6])
        >>> x_train = ds.array(x, (2, 2))
        >>> y_train = ds.array(y, (2, 1))
        >>> model = RandomForestRegressor(n_estimators=2, random_state=0)
        >>> model.fit(x_train, y_train)
        >>> model.save_model('/tmp/model')
        >>> loaded_model = RandomForestRegressor()
        >>> loaded_model.load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
        >>> model_pred = model.predict(x_test)
        >>> loaded_model_pred = loaded_model.predict(x_test)
        >>> assert np.allclose(model_pred.collect(),
        >>> loaded_model_pred.collect())
        """
        super().save_model(
            filepath,
            overwrite=overwrite,
            save_format=save_format
        )

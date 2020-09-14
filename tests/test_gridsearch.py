import unittest
import numpy as np

from sklearn import clone, datasets

import dislib as ds
from dislib.classification import CascadeSVM, RandomForestClassifier
from dislib.cluster import DBSCAN, KMeans, GaussianMixture
from dislib.decomposition import PCA
from dislib.neighbors import NearestNeighbors
from dislib.preprocessing import StandardScaler
from dislib.recommendation import ALS
from dislib.regression import LinearRegression
from dislib.model_selection import GridSearchCV, KFold
from dislib.utils import shuffle


class GridSearchCVTest(unittest.TestCase):

    def test_estimators_compatibility(self):
        """Tests that dislib estimators are compatible with GridSearchCV.

        GridSearchCV uses sklearn.clone(estimator), that requires estimators to
        have methods get_params() and set_params() working properly. This is
        what this test checks, and it can be easily achieved by making the
        estimators inherit from sklearn BaseEstimator"""
        estimators = (CascadeSVM, RandomForestClassifier,
                      DBSCAN, KMeans, GaussianMixture,
                      PCA, NearestNeighbors, ALS, LinearRegression)

        for estimator_class in estimators:
            self.assertIsInstance(estimator_class, type)
            est = estimator_class()
            # test __repr__
            repr(est)
            # test cloning
            cloned = clone(est)
            # test that set_params returns self
            self.assertIs(cloned.set_params(), cloned)
            # Checks if get_params(deep=False) is a subset of
            # get_params(deep=True)
            shallow_params = est.get_params(deep=False)
            deep_params = est.get_params(deep=True)
            self.assertTrue(all(item in deep_params.items()
                                for item in shallow_params.items()))

    def test_fit(self):
        """Tests GridSearchCV fit()."""
        x_np, y_np = datasets.load_iris(return_X_y=True)
        x = ds.array(x_np, (30, 4))
        y = ds.array(y_np[:, np.newaxis], (30, 1))

        param_grid = {'n_estimators': (2, 4),
                      'max_depth': range(3, 5)}
        rf = RandomForestClassifier()

        searcher = GridSearchCV(rf, param_grid)
        searcher.fit(x, y)

        expected_keys = {'param_max_depth', 'param_n_estimators', 'params',
                         'mean_test_score', 'std_test_score',
                         'rank_test_score'}
        split_keys = {'split%d_test_score' % i for i in range(5)}
        expected_keys.update(split_keys)
        self.assertSetEqual(set(searcher.cv_results_.keys()), expected_keys)

        expected_params = [(3, 2), (3, 4), (4, 2), (4, 4)]
        for params in searcher.cv_results_['params']:
            m = params['max_depth']
            n = params['n_estimators']
            self.assertIn((m, n), expected_params)
            expected_params.remove((m, n))
        self.assertEqual(len(expected_params), 0)

        self.assertTrue(hasattr(searcher, 'best_estimator_'))
        self.assertTrue(hasattr(searcher, 'best_score_'))
        self.assertTrue(hasattr(searcher, 'best_params_'))
        self.assertTrue(hasattr(searcher, 'best_index_'))
        self.assertTrue(hasattr(searcher, 'scorer_'))
        self.assertEqual(searcher.n_splits_, 5)

    def test_fit_2(self):
        """Tests GridSearchCV fit() with different data."""
        x_np, y_np = datasets.load_breast_cancer(return_X_y=True)
        x = ds.array(x_np, block_size=(100, 10))
        x = StandardScaler().fit_transform(x)
        y = ds.array(y_np.reshape(-1, 1), block_size=(100, 1))
        parameters = {'c': [0.1], 'gamma': [0.1]}
        csvm = CascadeSVM()
        searcher = GridSearchCV(csvm, parameters, cv=5)
        searcher.fit(x, y)

        self.assertTrue(hasattr(searcher, 'best_estimator_'))
        self.assertTrue(hasattr(searcher, 'best_score_'))
        self.assertTrue(hasattr(searcher, 'best_params_'))
        self.assertTrue(hasattr(searcher, 'best_index_'))
        self.assertTrue(hasattr(searcher, 'scorer_'))
        self.assertEqual(searcher.n_splits_, 5)

    def test_refit_false(self):
        """Tests GridSearchCV fit() with refit=False."""
        x_np, y_np = datasets.load_iris(return_X_y=True)
        x = ds.array(x_np, (30, 4))
        y = ds.array(y_np[:, np.newaxis], (30, 1))

        seed = 0
        x, y = shuffle(x, y, random_state=seed)

        param_grid = {'max_iter': range(1, 5)}
        csvm = CascadeSVM(check_convergence=False)
        searcher = GridSearchCV(csvm, param_grid, cv=3, refit=False)
        searcher.fit(x, y)

        self.assertFalse(hasattr(searcher, 'best_estimator_'))
        self.assertTrue(hasattr(searcher, 'best_score_'))
        self.assertTrue(hasattr(searcher, 'best_params_'))
        self.assertTrue(hasattr(searcher, 'best_index_'))
        self.assertTrue(hasattr(searcher, 'scorer_'))
        self.assertEqual(searcher.n_splits_, 3)

    def test_scoring_callable(self):
        """Tests GridSearchCV with callable scoring parameter."""
        x_np, y_np = datasets.load_iris(return_X_y=True)
        x = ds.array(x_np, (30, 4))
        y = ds.array(y_np[:, np.newaxis], (30, 1))

        param_grid = {'n_estimators': (2, 4)}
        rf = RandomForestClassifier()

        def scoring(clf, x_score, y_real):
            return clf.score(x_score, y_real)

        searcher = GridSearchCV(rf, param_grid, cv=3, scoring=scoring)
        searcher.fit(x, y)

        self.assertTrue(hasattr(searcher, 'cv_results_'))
        self.assertTrue(hasattr(searcher, 'best_estimator_'))
        self.assertTrue(hasattr(searcher, 'best_score_'))
        self.assertTrue(hasattr(searcher, 'best_params_'))
        self.assertTrue(hasattr(searcher, 'best_index_'))
        self.assertTrue(hasattr(searcher, 'scorer_'))

        def invalid_scoring(clf, x_score, y_score):
            return '2'

        searcher = GridSearchCV(rf, param_grid, cv=3, scoring=invalid_scoring)
        with self.assertRaisesRegex(ValueError,
                                    'scoring must return a number'):
            searcher.fit(x, y)

    def test_scoring_dict(self):
        """Tests GridSearchCV with scoring parameter of type dict."""
        x_np, y_np = datasets.load_iris(return_X_y=True)
        x = ds.array(x_np, (30, 4))
        y = ds.array(y_np[:, np.newaxis], (30, 1))

        param_grid = {'n_estimators': (2, 4)}
        rf = RandomForestClassifier()

        def hard_vote_score(rand_forest, x, y):
            rand_forest.hard_vote = True
            score = rand_forest.score(x, y)
            rand_forest.hard_vote = False
            return score

        scoring = {'default_score': None, 'custom_score': hard_vote_score}

        searcher = GridSearchCV(rf, param_grid, cv=3, scoring=scoring,
                                refit=False)
        searcher.fit(x, y)

        self.assertTrue(hasattr(searcher, 'cv_results_'))
        self.assertFalse(hasattr(searcher, 'best_estimator_'))
        self.assertFalse(hasattr(searcher, 'best_score_'))
        self.assertFalse(hasattr(searcher, 'best_params_'))
        self.assertFalse(hasattr(searcher, 'best_index_'))
        self.assertTrue(hasattr(searcher, 'scorer_'))

        searcher = GridSearchCV(rf, param_grid, cv=3, scoring=scoring,
                                refit=True)
        with self.assertRaises(ValueError):
            searcher.fit(x, y)

    def test_scoring_invalid(self):
        """Tests GridSearchCV raises error with invalid scoring parameter."""
        x_np, y_np = datasets.load_iris(return_X_y=True)
        x = ds.array(x_np, (30, 4))
        y = ds.array(y_np[:, np.newaxis], (30, 1))

        param_grid = {'n_estimators': (2, 4)}
        rf = RandomForestClassifier()

        searcher = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc',
                                refit=False)
        with self.assertRaises(ValueError):
            searcher.fit(x, y)

    def test_refit_callable(self):
        """Tests GridSearchCV with callable refit parameter."""
        x_np, y_np = datasets.load_iris(return_X_y=True)
        x = ds.array(x_np, (30, 4))
        y = ds.array(y_np[:, np.newaxis], (30, 1))
        param_grid = {'n_estimators': (2, 4)}
        rf = RandomForestClassifier()

        best_index = 1

        def refit(results):
            return best_index

        searcher = GridSearchCV(rf, param_grid, cv=3, refit=refit)
        searcher.fit(x, y)

        self.assertTrue(hasattr(searcher, 'cv_results_'))
        self.assertTrue(hasattr(searcher, 'best_estimator_'))
        self.assertFalse(hasattr(searcher, 'best_score_'))
        self.assertTrue(hasattr(searcher, 'best_params_'))
        self.assertTrue(hasattr(searcher, 'best_index_'))
        self.assertTrue(hasattr(searcher, 'scorer_'))

        best_index = 'str'
        searcher = GridSearchCV(rf, param_grid, cv=3, refit=refit)
        with self.assertRaises(TypeError):
            searcher.fit(x, y)

        best_index = -1
        searcher = GridSearchCV(rf, param_grid, cv=3, refit=refit)
        with self.assertRaises(IndexError):
            searcher.fit(x, y)

    def test_param_grid_invalid(self):
        """Tests GridSearchCV with invalid param_grid parameter."""
        x_np, y_np = datasets.load_iris(return_X_y=True)
        x = ds.array(x_np, (30, 4))
        y = ds.array(y_np[:, np.newaxis], (30, 1))
        rf = RandomForestClassifier()
        param_grid = {'n_estimators': np.array([[1, 2], [3, 4]])}
        with self.assertRaises(ValueError):
            searcher = GridSearchCV(rf, param_grid, cv=3)
            searcher.fit(x, y)
        param_grid = {'n_estimators': '2'}
        with self.assertRaises(ValueError):
            searcher = GridSearchCV(rf, param_grid, cv=3)
            searcher.fit(x, y)
        param_grid = {'n_estimators': []}
        with self.assertRaises(ValueError):
            searcher = GridSearchCV(rf, param_grid, cv=3)
            searcher.fit(x, y)

    def test_cv_class(self):
        """Tests GridSearchCV with a class cv parameter."""
        x_np, y_np = datasets.load_iris(return_X_y=True)
        x = ds.array(x_np, (30, 4))
        y = ds.array(y_np[:, np.newaxis], (30, 1))
        rf = RandomForestClassifier()
        param_grid = {'n_estimators': (2, 4)}
        searcher = GridSearchCV(rf, param_grid, cv=KFold(4))
        searcher.fit(x, y)

        self.assertTrue(hasattr(searcher, 'cv_results_'))
        self.assertTrue(hasattr(searcher, 'best_estimator_'))
        self.assertTrue(hasattr(searcher, 'best_score_'))
        self.assertTrue(hasattr(searcher, 'best_params_'))
        self.assertTrue(hasattr(searcher, 'best_index_'))
        self.assertTrue(hasattr(searcher, 'scorer_'))

    def test_cv_invalid(self):
        """Tests GridSearchCV with invalid cv parameter."""
        x_np, y_np = datasets.load_iris(return_X_y=True)
        x = ds.array(x_np, (30, 4))
        y = ds.array(y_np[:, np.newaxis], (30, 1))
        rf = RandomForestClassifier()
        param_grid = {'n_estimators': (2, 4)}
        with self.assertRaises(ValueError):
            searcher = GridSearchCV(rf, param_grid, cv={})
            searcher.fit(x, y)

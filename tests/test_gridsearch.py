import unittest
import numpy as np

from sklearn import datasets

from dislib.classification import RandomForestClassifier, CascadeSVM
from dislib.data import load_data
from dislib.model_selection import GridSearchCV, KFold


class GridSearchCVTest(unittest.TestCase):

    def test_fit(self):
        """Tests GridSearchCV fit()."""
        x, y = datasets.load_iris(return_X_y=True)
        ds = load_data(x, 30, y)

        param_grid = {'n_estimators': (2, 4),
                      'max_depth': range(3, 5)}
        rf = RandomForestClassifier()

        searcher = GridSearchCV(rf, param_grid)
        searcher.fit(ds)

        expected_keys = {'param_max_depth', 'param_n_estimators', 'params',
                         'split0_test_score', 'split1_test_score',
                         'split2_test_score', 'split3_test_score',
                         'split4_test_score', 'mean_test_score',
                         'std_test_score', 'rank_test_score',
                         'std_master_fit_time', 'mean_master_fit_time',
                         'std_master_score_time', 'mean_master_score_time'}
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

    def test_refit_false(self):
        """Tests GridSearchCV fit() with refit=False."""
        x, y = datasets.load_iris(return_X_y=True)

        ds = load_data(x, 30, y)

        param_grid = {'max_iter': range(1, 5)}
        csvm = CascadeSVM(check_convergence=False)
        searcher = GridSearchCV(csvm, param_grid, cv=3, refit=False)
        searcher.fit(ds)

        self.assertFalse(hasattr(searcher, 'best_estimator_'))
        self.assertTrue(hasattr(searcher, 'best_score_'))
        self.assertTrue(hasattr(searcher, 'best_params_'))
        self.assertTrue(hasattr(searcher, 'best_index_'))
        self.assertTrue(hasattr(searcher, 'scorer_'))
        self.assertEqual(searcher.n_splits_, 3)

    def test_scoring_callable(self):
        """Tests GridSearchCV with callable scoring parameter."""
        x, y = datasets.load_iris(return_X_y=True)
        ds = load_data(x, 30, y)

        param_grid = {'n_estimators': (2, 4)}
        rf = RandomForestClassifier()

        def scoring(clf, pds):
            return clf.score(pds)

        searcher = GridSearchCV(rf, param_grid, cv=3, scoring=scoring)
        searcher.fit(ds)

        self.assertTrue(hasattr(searcher, 'cv_results_'))
        self.assertTrue(hasattr(searcher, 'best_estimator_'))
        self.assertTrue(hasattr(searcher, 'best_score_'))
        self.assertTrue(hasattr(searcher, 'best_params_'))
        self.assertTrue(hasattr(searcher, 'best_index_'))
        self.assertTrue(hasattr(searcher, 'scorer_'))

        def invalid_scoring(clf, pds):
            return '2'

        searcher = GridSearchCV(rf, param_grid, cv=3, scoring=invalid_scoring)
        with self.assertRaisesRegex(ValueError,
                                    'scoring must return a number'):
            searcher.fit(ds)

    def test_scoring_dict(self):
        """Tests GridSearchCV with scoring parameter of type dict."""
        x, y = datasets.load_iris(return_X_y=True)
        ds = load_data(x, 30, y)

        param_grid = {'n_estimators': (2, 4)}
        rf = RandomForestClassifier()

        def hard_vote_score(rand_forest, dataset):
            rand_forest.hard_vote = True
            score = rand_forest.score(dataset)
            rand_forest.hard_vote = False
            return score

        scoring = {'default_score': None, 'custom_score': hard_vote_score}

        searcher = GridSearchCV(rf, param_grid, cv=3, scoring=scoring,
                                refit=False)
        searcher.fit(ds)

        self.assertTrue(hasattr(searcher, 'cv_results_'))
        self.assertFalse(hasattr(searcher, 'best_estimator_'))
        self.assertFalse(hasattr(searcher, 'best_score_'))
        self.assertFalse(hasattr(searcher, 'best_params_'))
        self.assertFalse(hasattr(searcher, 'best_index_'))
        self.assertTrue(hasattr(searcher, 'scorer_'))

        searcher = GridSearchCV(rf, param_grid, cv=3, scoring=scoring,
                                refit=True)
        with self.assertRaises(ValueError):
            searcher.fit(ds)

    def test_scoring_invalid(self):
        """Checks GridSearchCV raises error with invalid scoring parameter."""
        x, y = datasets.load_iris(return_X_y=True)
        ds = load_data(x, 30, y)

        param_grid = {'n_estimators': (2, 4)}
        rf = RandomForestClassifier()

        searcher = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc',
                                refit=False)
        with self.assertRaises(ValueError):
            searcher.fit(ds)

    def test_refit_callable(self):
        """Tests GridSearchCV with callable refit parameter."""
        x, y = datasets.load_iris(return_X_y=True)
        ds = load_data(x, 30, y)
        param_grid = {'n_estimators': (2, 4)}
        rf = RandomForestClassifier()

        best_index = 1

        def refit(results):
            return best_index

        searcher = GridSearchCV(rf, param_grid, cv=3, refit=refit)
        searcher.fit(ds)

        self.assertTrue(hasattr(searcher, 'cv_results_'))
        self.assertTrue(hasattr(searcher, 'best_estimator_'))
        self.assertFalse(hasattr(searcher, 'best_score_'))
        self.assertTrue(hasattr(searcher, 'best_params_'))
        self.assertTrue(hasattr(searcher, 'best_index_'))
        self.assertTrue(hasattr(searcher, 'scorer_'))

        best_index = 'str'
        searcher = GridSearchCV(rf, param_grid, cv=3, refit=refit)
        with self.assertRaises(TypeError):
            searcher.fit(ds)

        best_index = -1
        searcher = GridSearchCV(rf, param_grid, cv=3, refit=refit)
        with self.assertRaises(IndexError):
            searcher.fit(ds)

    def test_param_grid_invalid(self):
        """Tests GridSearchCV with invalid param_grid parameter."""
        x, y = datasets.load_iris(return_X_y=True)
        ds = load_data(x, 30, y)
        rf = RandomForestClassifier()
        param_grid = {'n_estimators': np.array([[1, 2], [3, 4]])}
        with self.assertRaises(ValueError):
            searcher = GridSearchCV(rf, param_grid, cv=3)
            searcher.fit(ds)
        param_grid = {'n_estimators': '2'}
        with self.assertRaises(ValueError):
            searcher = GridSearchCV(rf, param_grid, cv=3)
            searcher.fit(ds)
        param_grid = {'n_estimators': []}
        with self.assertRaises(ValueError):
            searcher = GridSearchCV(rf, param_grid, cv=3)
            searcher.fit(ds)

    def test_cv_class(self):
        """Tests GridSearchCV with a class cv parameter."""
        x, y = datasets.load_iris(return_X_y=True)
        ds = load_data(x, 30, y)
        rf = RandomForestClassifier()
        param_grid = {'n_estimators': (2, 4)}
        searcher = GridSearchCV(rf, param_grid, cv=KFold(4))
        searcher.fit(ds)

        self.assertTrue(hasattr(searcher, 'cv_results_'))
        self.assertTrue(hasattr(searcher, 'best_estimator_'))
        self.assertTrue(hasattr(searcher, 'best_score_'))
        self.assertTrue(hasattr(searcher, 'best_params_'))
        self.assertTrue(hasattr(searcher, 'best_index_'))
        self.assertTrue(hasattr(searcher, 'scorer_'))

    def test_cv_invalid(self):
        """Tests GridSearchCV with invalid cv parameter."""
        x, y = datasets.load_iris(return_X_y=True)
        ds = load_data(x, 30, y)
        rf = RandomForestClassifier()
        param_grid = {'n_estimators': (2, 4)}
        with self.assertRaises(ValueError):
            searcher = GridSearchCV(rf, param_grid, cv={})
            searcher.fit(ds)

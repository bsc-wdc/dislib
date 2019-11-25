import unittest
import scipy.stats as stats
import numpy as np

from sklearn import datasets

import dislib as ds
from dislib.classification import CascadeSVM
from dislib.model_selection import RandomizedSearchCV


class RandomizedSearchCVTest(unittest.TestCase):
    def test_fit(self):
        """Tests RandomizedSearchCV fit()."""
        x_np, y_np = datasets.load_iris(return_X_y=True)
        p = np.random.permutation(len(x_np))  # Pre-shuffling required for CSVM
        x = ds.array(x_np[p], (30, 4))
        y = ds.array((y_np[p] == 0)[:, np.newaxis], (30, 1))
        param_distributions = {'c': stats.expon(scale=0.5),
                               'gamma': stats.expon(scale=1)}
        csvm = CascadeSVM()
        n_iter = 12
        k = 3
        searcher = RandomizedSearchCV(estimator=csvm,
                                      param_distributions=param_distributions,
                                      n_iter=n_iter, cv=k, random_state=0)
        searcher.fit(x, y)

        expected_keys = {'param_c', 'param_gamma', 'params', 'mean_test_score',
                         'std_test_score', 'rank_test_score'}
        split_keys = {'split%d_test_score' % i for i in range(k)}
        expected_keys.update(split_keys)

        self.assertSetEqual(set(searcher.cv_results_.keys()), expected_keys)
        self.assertEqual(len(searcher.cv_results_['param_c']), n_iter)
        self.assertTrue(hasattr(searcher, 'best_estimator_'))
        self.assertTrue(hasattr(searcher, 'best_score_'))
        self.assertTrue(hasattr(searcher, 'best_params_'))
        self.assertTrue(hasattr(searcher, 'best_index_'))
        self.assertTrue(hasattr(searcher, 'scorer_'))
        self.assertEqual(searcher.n_splits_, k)

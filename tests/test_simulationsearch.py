from dislib.model_selection import SimulationGridSearch
from tests import BaseTimedTestCase
import pandas as pd


def my_simulation(a, b):
    return (a*a)/(b*b)+a*(a+b)-b*(2*b)


class SimulationSearchCVTest(BaseTimedTestCase):

    def test_fit(self):
        param_grid = {'a': [-1.1, -0.1, 1.5, 2.5], 'b': [0.1, 1.5, 2.5, 3.5]}
        searcher = SimulationGridSearch(my_simulation, param_grid, order="min")
        searcher.fit(None)
        expected_keys = {'param_a', 'param_b', 'params',
                         'mean_test_simulation', 'std_test_simulation',
                         'rank_test_simulation'}
        split_keys = {'results_%d_test_simulation' % i for i in range(1)}
        expected_keys.update(split_keys)
        self.assertSetEqual(set(searcher.cv_results_.keys()), expected_keys)
        self.assertTrue(hasattr(searcher, 'best_score_'))
        self.assertTrue(hasattr(searcher, 'best_params_'))
        self.assertTrue(hasattr(searcher, 'best_index_'))
        self.assertTrue(isinstance(searcher.raw_results, list))
        self.assertTrue(isinstance(searcher.raw_results[0], list))
        self.assertTrue(isinstance(searcher.raw_results[0][0], float))
        results_in_pandas = pd.DataFrame(searcher.cv_results_)
        self.assertTrue(isinstance(results_in_pandas, pd.DataFrame))

        param_grid = {'a': [-1.1, -0.1, 1.5, 2.5], 'b': [0.1, 1.5, 2.5, 3.5]}
        searcher_max = SimulationGridSearch(my_simulation,
                                            param_grid, order="max")
        searcher_max.fit(None)
        expected_keys = {'param_a', 'param_b', 'params',
                         'mean_test_simulation', 'std_test_simulation',
                         'rank_test_simulation'}
        split_keys = {'results_%d_test_simulation' % i for i in range(1)}
        expected_keys.update(split_keys)
        self.assertSetEqual(set(searcher.cv_results_.keys()), expected_keys)
        self.assertTrue(hasattr(searcher, 'best_score_'))
        self.assertTrue(hasattr(searcher, 'best_params_'))
        self.assertTrue(hasattr(searcher, 'best_index_'))
        results_max_in_pandas = pd.DataFrame(searcher_max.cv_results_)
        self.assertFalse(results_in_pandas['rank_test_simulation'].equals(
            results_max_in_pandas['rank_test_simulation']))

    def test_exception(self):
        param_grid = {'a': [-1.1, -0.1, 1.5, 2.5], 'b': [0.1, 1.5, 2.5, 3.5]}
        my_simulation_not_callable = 0
        searcher = SimulationGridSearch(my_simulation_not_callable,
                                        param_grid, order="min")
        with self.assertRaises(NotImplementedError):
            searcher.fit(None)

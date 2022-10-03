from pycompss.api.api import compss_wait_on
from sklearn.model_selection import ParameterGrid
from dislib.model_selection._validation import simulation_execution
from collections import defaultdict
from functools import partial
from numpy.ma import MaskedArray
from scipy.stats import rankdata
import numpy as np


class SimulationGridSearch():
    def __init__(self, estimator, param_grid,
                 sim_number=1, order="max"):
        self.estimator = estimator
        self.param_grid = param_grid
        self.sim_number = sim_number
        self.order = order
        self.raw_results = None
        self.cv_results_ = None

    def _run_search(self, evaluate_candidates):
        """Abstract method to perform the search. The parameter
        `evaluate_candidates` is a function that evaluates a ParameterGrid at a
        time """
        evaluate_candidates(ParameterGrid(self.param_grid))

    def fit(self, x, y=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        x : ds-array
            Training data samples.
        y : ds-array, optional (default = None)
            Training data labels or values.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator
        all_candidate_params = []
        all_out = []

        def evaluate_candidates_simulation(candidate_params):
            candidate_params = list(candidate_params)
            fits = []
            for parameters in candidate_params:
                fits.append(simulation_execution(
                    estimator, parameters=parameters,
                    simulation_params=fit_params,
                    number_simulations=self.sim_number))
            out = [simulation_results for simulation_results in fits]
            out = compss_wait_on(out)
            all_candidate_params.extend(candidate_params)
            all_out.extend(out)

        if callable(estimator):
            self._run_search(evaluate_candidates_simulation)
        else:
            raise NotImplementedError("The simulation needs to "
                                      "be contained on a function")

        self.raw_results = all_out
        results = self._format_results(all_candidate_params, all_out,
                                       order=self.order,
                                       sim_number=self.sim_number)

        self.best_index_ = results["rank_test_simulation"].argmin()
        self.best_score_ = results["mean_test_simulation"][self.best_index_]
        self.best_params_ = results["params"][self.best_index_]

        self.cv_results_ = results

        return self

    @staticmethod
    def _format_results(candidate_params, out, order="max", sim_number=1):
        n_candidates = len(candidate_params)
        test_scores = out
        results = {}

        def _store(key_name, array, rank=False):
            """A small helper to store the scores/times to
            the cv_results_"""
            array = np.array(array, dtype=np.float64).reshape(
                n_candidates, sim_number)
            if sim_number > 0:
                for i in range(sim_number):
                    results["results_%d_%s" % (i, key_name)] = array[:, i]
            array_means = np.mean(array, axis=1)
            array_stds = np.std(array, axis=1)
            results['mean_%s' % key_name] = array_means
            results['std_%s' % key_name] = array_stds
            if rank:
                if order == "max":
                    results["rank_%s" % key_name] = np.asarray(
                        rankdata(-array_means, method='min'), dtype=np.int32)
                else:
                    results["rank_%s" % key_name] = np.asarray(
                        rankdata(array_means, method='min'), dtype=np.int32)
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates, ),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params
        _store('test_simulation', test_scores, rank=True)
        return results

from pycompss.api.api import compss_wait_on
from sklearn.model_selection import ParameterGrid
from dislib.model_selection._validation import simulation_execution
from collections import defaultdict
from functools import partial
from numpy.ma import MaskedArray
from scipy.stats import rankdata
import numpy as np


class SimulationGridSearch:
    """Exhaustive execution of all combinations of specified parameters values
     in parallel simulations.

    SimulationGridSearch implements a "fit" method.

    Parameters
    ----------
    estimator : simulator object.
        This should receive the parameters specified in param_grid and use
        that parameters for the corresponding operation.
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    sim_number : Integer
        Number of simulations that are going to be executed with each of the
        parameter combination.
    order : string "max" or "min".
        String that specifies how to order the results obtained from
        the simulation, "max" will set first the highest values
        and "min" the lowest values.

    Examples
    --------
    >>> import dislib as ds
    >>> from dislib.model_selection import SimulationGridSearch
    >>> from dislib.classification import RandomForestClassifier
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> def my_simulation(a, b):
    >>>    return (a*a)/(b*b)+a*(a+b)-b*(2*b)
    >>>
    >>> param_grid = {'a': [-1.1, -0.1, 1.5, 2.5], 'b': [0.1, 1.5, 2.5, 3.5]}
    >>> searcher = SimulationGridSearch(my_simulation, param_grid, order="min")
    >>> searcher.fit(None)
    >>> best_params = searcher.best_params_
    >>>

    Attributes
    ----------
    raw_results : list of objects
        List containing the results obtained from the different simulations.
        In the list the results are saved as returned from the simulation,
        with no changes in the format.
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.
        For instance the below given table:

        +------------+------------+-----------------+---+---------+
        |   param_a  |   param_b  |        params       |...|rank_t...|
        +============+============+=================+===+=========+
        |    -1.1    |     0.1    |{'a': -1.1, 'b': 0.1}|...|    2    |
        +------------+------------+-----------------+---+---------+
        |    -1.1    |     1.5    |{'a': -1.1, 'b': 1.5}|...|    4    |
        +------------+------------+-----------------+---+---------+
        |    -0.1    |     0.1    |{'a': -0.1, 'b': 0.1}|...|    3    |
        +------------+------------+-----------------+---+---------+
        |    -0.1    |     1.5    |{'a': -0.1, 'b': 1.5}|...|    1    |
        +------------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = [-1.1, -1.1, -0.1, -0.1],
                                         mask = [False False False False]...),
            'param_degree': masked_array(data = [0.1 1.5 0.1 1.5],
                                         mask = [False False  False  False]
                                         ...),
            ...
            'mean_test_simulation'    : [122.08, -4.40, 0.98, -4.63],
            'std_test_simulation'     : [0.0, --, --, --],
            'rank_test_score'    : [2, 4, 3, 1],
            'params'             : [{'a': '-1.1', 'b': 0.1}, ...],
            }

        NOTES:

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter used in the simulation.

    best_score_ : float
        Best value obtained from a simulation, if several runs of each
        simulation are done the best mean of the values obtained is used
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.
        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).
    """
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

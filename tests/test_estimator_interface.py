import sys
import unittest
from sklearn.utils.estimator_checks import \
    check_parameters_default_constructible, check_no_attributes_set_in_init, \
    check_get_params_invariance

from dislib.classification import CascadeSVM, RandomForestClassifier
from dislib.cluster import DBSCAN, KMeans, GaussianMixture
from dislib.decomposition import PCA
from dislib.neighbors import NearestNeighbors
from dislib.recommendation import ALS
from dislib.regression import LinearRegression


class EstimatorsInterfaceTest(unittest.TestCase):

    def test_estimators_interface(self):
        """Checks that dislib estimators adhere to a common interface based on
        scikit-learn conventions that can help developing global features.

        Only some checks taken from `sklearn.utils.estimator_checks.
        check_estimator()` are included. Other checks, validating that fit()
        and other estimator methods work as expected, cannot be included
        directly due to interface differences with scikit-learn, but could be
        developed ad hoc."""
        estimators = (CascadeSVM, RandomForestClassifier,
                      DBSCAN, KMeans, GaussianMixture,
                      PCA, NearestNeighbors, ALS, LinearRegression)

        clean_invalid_modules()  # Workaround to an issue with sklearn

        for estimator in estimators:
            self.assertTrue(check_estimator(estimator))


def check_estimator(estimator_class):
    assert isinstance(estimator_class, type)
    name = estimator_class.__name__
    estimator = estimator_class()
    check_parameters_default_constructible(name, estimator_class)
    check_no_attributes_set_in_init(name, estimator)
    check_get_params_invariance(name, estimator)
    return True


def clean_invalid_modules():
    reg = "__warningregistry__"
    for mod_name, mod in list(sys.modules.items()):
        if hasattr(mod, reg) and getattr(mod, reg) is None:
            del sys.modules[mod_name]

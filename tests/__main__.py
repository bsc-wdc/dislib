import unittest
from concurrent.futures import ThreadPoolExecutor

T = [["test_lasso"],
     ["test_array", "test_pca", "test_daura"],
     ["test_gm", "test_preproc", "test_decision_tree"],
     ["test_qr", "test_kmeans", "test_knn"],
     ["test_gridsearch", "test_tsqr", "test_linear_regression"],
     ["test_dbscan", "test_matmul", "test_als"],
     ["test_rf_classifier", "test_randomizedsearch",
      "test_data_utils", "test_kfold"],
     ["test_csvm", "test_rf_regressor", "test_utils", "test_rf_dataset"]]


def run_tests(tests_to_run):
    test_suite = unittest.TestSuite()
    test_suite.addTests(tests_to_run)
    unittest.TextTestRunner(verbosity=2).run(test_suite)


if __name__ == '__main__':
    suite = list(unittest.loader.defaultTestLoader.discover('./tests/'))

    tests_to_run = []
    for tests in T:
        ttt = []
        for t in tests:
            for test_case in suite:
                if t.lower() in str(test_case).lower():
                    ttt.append(test_case)
        tests_to_run.append(ttt)

    with ThreadPoolExecutor(max_workers=16) as exec:
        exec.map(run_tests, tests_to_run)

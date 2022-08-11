import unittest
import argparse


if __name__ == '__main__':
    suite = list(unittest.loader.defaultTestLoader.discover('./tests/'))
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('tests', metavar='T', type=str, nargs='+',
                        help='an integer for the accumulator')

    args = parser.parse_args()
    tests_to_run = []
    for t in args.tests:
        for test_case in suite:
            if t.lower() in str(test_case).lower():
                tests_to_run.append(test_case)

    print('********** TESTS TO RUN *************')
    print(tests_to_run)
    print('*************************************')

    test_suite = unittest.TestSuite()
    test_suite.addTests(tests_to_run)
    unittest.TextTestRunner(verbosity=2).run(test_suite)

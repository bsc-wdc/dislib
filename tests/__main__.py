import unittest
import argparse
import datetime

if __name__ == '__main__':
    suite = list(unittest.loader.defaultTestLoader.discover('./tests/'))
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('tests', metavar='T', type=str, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument('-id', type=int)

    args = parser.parse_args()
    tests_to_run = []
    for t in args.tests:
        for test_case in suite:
            if t.lower() in str(test_case).lower():
                tests_to_run.append(test_case)

    print(f'WORKER {args.id} START TEST AT', datetime.datetime.now())

    test_suite = unittest.TestSuite()
    test_suite.addTests(tests_to_run)
    unittest.TextTestRunner(verbosity=2).run(test_suite)

    print(f'WORKER {args.id} END TEST AT', datetime.datetime.now())

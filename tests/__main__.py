import unittest
import threading


def load_tests(loader):
    return loader.discover('./tests/')


def run_test(test):
    suite = unittest.TestSuite()
    suite.addTest(test)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    suite = load_tests(unittest.loader.defaultTestLoader)
    threads = []
    for case in suite:
        thread = threading.Thread(target=run_test, args=(case,))
        thread.start()
        threads.append(thread)

    [t.join() for t in threads]

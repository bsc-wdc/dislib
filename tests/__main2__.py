import unittest


def load_tests(loader, tests, pattern):
    return loader.discover('./tests/')


if __name__ == '__main__':
    unittest.main(verbosity=2)

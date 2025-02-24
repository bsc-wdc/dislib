import performance

import dislib as ds
from dislib.utils import shuffle


def main():
    x = ds.random_array((20000, 20000), (100, 100))
    y = ds.random_array((20000, 1), (100, 1))
    performance.measure("Shuf", "20K", shuffle, x, y, 25)


if __name__ == "__main__":
    main()

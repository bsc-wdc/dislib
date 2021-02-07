import performance

import dislib as ds


def main():
    x = ds.random_array((20000, 20000), (100, 100))
    performance.measure("TR", "20K", x.transpose)


if __name__ == "__main__":
    main()

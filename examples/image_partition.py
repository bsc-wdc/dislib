from math import sqrt, ceil
from itertools import product, chain
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from os.path import splitext, isfile, basename, join
from dislib.cluster import KMeans, DBSCAN, GaussianMixture
import dislib as ds

# == START algorithm definitions ==
# Add new algorithms to `available_algorithms` and to the following functions.
# Also, new parameters have to be added to the parser further below.
# Algorithms must have a fit_predict() method.
available_algorithms = 'KMeans', 'GaussianMixture', 'DBSCAN'


def get_kwargs(args, alg_args_equivalence):
    args = vars(args)
    kwargs = {}
    for alg_arg_name, args_arg_name in alg_args_equivalence:
        if args_arg_name in args:
            kwargs[alg_arg_name] = args[args_arg_name]
    return kwargs


def get_kmeans_kwargs(args):
    return get_kwargs(args,
                      {('n_clusters', 'n_clusters'),
                       ('random_state', 'random_seed'),
                       ('max_iter', 'max_iter'),
                       ('tol', 'tol')})


def get_gm_kwargs(args):
    return get_kwargs(args,
                      {('n_components', 'n_clusters'),
                       ('random_state', 'random_seed'),
                       ('max_iter', 'max_iter'),
                       ('tol', 'tol')})


def get_dbscan_kwargs(args):
    return get_kwargs(args,
                      {('eps', 'eps'),
                       ('min_samples', 'min_samples'),
                       ('n_regions', 'n_regions'),
                       ('max_samples', 'max_samples')})


def initialize(alg_names, args):
    return [{
        'KMeans': lambda x: KMeans(**get_kmeans_kwargs(x)),
        'DBSCAN': lambda x: DBSCAN(**get_dbscan_kwargs(x)),
        'GaussianMixture': lambda x: GaussianMixture(**get_gm_kwargs(x))
        }[name](args) for name in alg_names]


def save_as_params(alg_name, args):
    return {
        'KMeans': lambda x: 'n' + str(x.n_clusters),
        'DBSCAN': lambda x: 'e' + str(x.eps) + 'm' + str(x.min_samples),
        'GaussianMixture': lambda x: 'n' + str(x.n_clusters)
        }[alg_name](args)


def get_noise_value(alg_name):
    return {
        'DBSCAN': -1
        }.get(alg_name, None)


def execution_summary(alg_name, algorithm):
    return {
        'KMeans': lambda x: "KMeans iterations: " + str(x.n_iter),
        'DBSCAN': lambda x: "Number of clusters found: " + str(x.n_clusters),
        'GaussianMixture':
        lambda x: "GaussianMixture iterations: " + str(x.n_iter)
        }[alg_name](algorithm)

# == END algorithm definitions ==


def main():
    parser = argparse.ArgumentParser(description="Partition an image using "
                                                 "dislib clustering "
                                                 "algorithms.")
    # Positional arguments
    parser.add_argument('img', metavar='IMAGE',
                        help="image to partition (.png format preferable)")
    # Optional arguments
    parser.add_argument('-a', '--algorithm', choices=available_algorithms,
                        action='append',
                        help="clustering algorithm, this argument can be "
                             "provided multiple times (by default all "
                             "available algorithms are used)")
    parser.add_argument('-n', '--n_clusters', type=int, default=10,
                        help="number of clusters used by KMeans or Gaussian "
                             "Mixture to partition the image (default is 10)")
    parser.add_argument('-i', '--max_iter', type=int,
                        default=argparse.SUPPRESS,
                        help="see the docs for dislib.cluster.KMeans and "
                             "dislib.cluster.GaussianMixture (defaults are "
                             "kept)")
    parser.add_argument('-t', '--tol', type=float, default=argparse.SUPPRESS,
                        help="see the docs for dislib.cluster.KMeans and "
                             "dislib.cluster.GaussianMixture (defaults are "
                             "kept)")
    parser.add_argument('-e', '--eps', type=float, default=10.,
                        help="see dislib.cluster.DBSCAN docs (default is 10.)")
    parser.add_argument('-m', '--min_samples', type=int, default=35,
                        help="see dislib.cluster.DBSCAN docs (default is 35)")
    parser.add_argument('-M', '--max_samples', type=int,
                        default=argparse.SUPPRESS,
                        help="see dislib.cluster.DBSCAN docs (default is "
                             "kept)")
    parser.add_argument('-N', '--n_regions', type=int,
                        default=argparse.SUPPRESS,
                        help="(not to confuse with --n_clusters)\n"
                             "see dislib.cluster.DBSCAN docs (default is "
                             "kept)")
    parser.add_argument('-c', '--chunks', type=int, default=50,
                        help="(not to confuse with --n_clusters)\n"
                             "number of chunks the dataset should be divided "
                             "for the distributed computation (default is 50)")
    parser.add_argument('-r', '--random_seed', type=int,
                        default=argparse.SUPPRESS,
                        help="seed for the random_state parameter of the "
                             "algorithms")
    parser.add_argument('-s', '--save', action='store_true',
                        help="save the partitioned images")
    parser.add_argument('-p', '--plot', action='store_true',
                        help="show (plot) the original and the partitioned "
                             "images in a matplotlib.pyplot window")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="print info messages")
    args = parser.parse_args()
    if args.verbose:
        print('Arguments: ', vars(args))

    if not args.save and not args.plot:
        print("WARNING: not using --save nor --plot arguments, the "
              "partitioned image(s) will be lost")

    if not isfile(args.img):
        raise FileNotFoundError("argument IMAGE: file not found:"
                                "'{}'".format(args.img))
    if args.algorithm is None:
        args.algorithm = available_algorithms
    algorithms = initialize(args.algorithm, args)

    img = mpimg.imread(args.img)

    if args.save:
        img_basename = basename(args.img)
        img_name, extension = splitext(img_basename)

    if args.plot:
        plot_num = 1
        n_images = len(algorithms) + 1
        dim_1 = ceil(sqrt(n_images-0.5))
        dim_2 = ceil((n_images-0.5)/dim_1)
        plt.subplot(dim_1, dim_2, plot_num)
        cmap = 'gray' if len(img.shape) == 2 else None
        plt.imshow(img, cmap=cmap)
        plot_num += 1

    if len(img.shape) == 3 and img.shape[2] == 4:
        print("WARNING: the image contains an alpha channel (transparency "
              "data) which is discarded for partitioning")
        img = img[:, :, 0:3]

    for alg_name, alg in zip(args.algorithm, algorithms):
        if args.verbose:
            print('Starting partition with algorithm:', alg_name)
        noise_value = get_noise_value(alg_name)
        t0 = time.time()
        part_img = image_partition(img, args.chunks, alg, noise_value)
        t1 = time.time()
        if args.verbose:
            print(execution_summary(alg_name, alg))
            print('Partition finished in {:0.2f} s.'.format(t1 - t0))
        if args.plot:
            plt.subplot(dim_1, dim_2, plot_num)
            plt.imshow(part_img)
            plt.title(alg_name, size=18)
            plot_num += 1

        if args.save:
            alg_params = save_as_params(alg_name, args)
            save_as = join(os.getcwd(),
                           img_name + alg_name + alg_params + '.png')
            mpimg.imsave(save_as, part_img)
            if args.verbose:
                print('Partitioned image saved at:', save_as)
    if args.plot:
        if args.verbose:
            print('Showing images in a new window...')
        plt.show()


def image_partition(img, n_chunks, algorithm, noise_value=None):
    if len(img.shape) == 3 and img.shape[2] == 4:
        print("WARNING: the image contains an alpha channel (transparency "
              "data) which is discarded for partitioning")
        img = img[:, :, 0:3]
    gray_scaled = len(img.shape) == 2
    n_channels = 1 if gray_scaled else img.shape[2]
    dims = 2 + n_channels
    height = img.shape[0]
    width = img.shape[1]
    iter_pixels = product(range(height), range(width))
    n_pixels = height * width
    img_data = np.empty((n_pixels, dims), dtype=np.float32)
    img_data[:, 0:2] = np.fromiter(chain.from_iterable(iter_pixels),
                                   dtype=np.float32).reshape(n_pixels, 2)
    if img.dtype == np.uint8:  # Channels expected to range between 0 and 255
        rescaling_factor = sqrt(n_pixels) / 255
    else:  # Channels expected to range between 0 and 1
        rescaling_factor = sqrt(n_pixels)
    img_data[:, 2:dims] = img.reshape(n_pixels, n_channels) * rescaling_factor
    x = ds.array(img_data, (ceil(n_pixels / n_chunks), dims))

    labels = algorithm.fit_predict(x).collect().astype(int)

    colors = plt.get_cmap('Set3').colors
    n_colors = len(colors)
    if noise_value:
        colors = ((0., 0., 0.),) + colors
        noise = labels == noise_value
        labels = labels % n_colors + 1
        labels[noise] = 0
    else:
        labels = labels % n_colors
    colors = np.array(colors, dtype=np.float32)[:, 0:3]

    return colors[labels].reshape(height, width, 3)


if __name__ == "__main__":
    main()

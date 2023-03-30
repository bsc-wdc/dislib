import os

from dislib.data.array import random_array, apply_along_axis, array, zeros, \
    full, identity, eye, matmul, concat_rows, concat_columns
from dislib.data.io import load_svmlight_file, load_npy_file, load_txt_file, \
    load_mdcrd_file, save_txt
from dislib.math import kron, svd

name = "dislib"
version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '../VERSION')

# If version file exists, we are being imported from sources, otherwise the
# package's being installed with setup.py
if os.path.isfile(version_file):
    # add the .dev to the version to differentiate importing sources from
    # installed packages
    __version__ = open(version_file).read().strip() + ".dev"
else:
    # only available when installed with setup.py
    try:
        import pkg_resources

        __version__ = pkg_resources.require("dislib")[0].version
    except Exception as e:
        print("This dislib installation does not have a version number. "
              "Probably it was not installed with setup.py.\n%s" % e)
        __version__ = 'unknown'

__all__ = ['array', 'random_array', 'zeros', 'full', 'identity', 'eye',
           'load_txt_file', 'load_svmlight_file', 'load_npy_file',
           'load_mdcrd_file', 'matmul', 'save_txt', 'concat_rows',
           'concat_columns', 'apply_along_axis', 'kron', 'svd']

gpu_envar = os.environ.get('DISLIB_GPU_AVAILABLE', 'False')
__gpu_available__ = gpu_envar.lower() == 'true'

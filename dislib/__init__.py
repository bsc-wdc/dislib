import os

from dislib.data.array import random_array, apply_along_axis, array, zeros, \
    full, identity
from dislib.data.io import load_svmlight_file, load_npy_file, load_txt_file
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

__all__ = ['array', 'random_array', 'zeros', 'full', 'identity',
           'load_txt_file', 'load_svmlight_file', 'load_npy_file',
           'apply_along_axis', 'kron', 'svd']

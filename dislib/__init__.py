import os

from dislib.data.array import random_array, apply_along_axis, array, zeros, \
    full, identity, eye, matmul, concat_rows, concat_columns, matadd, \
    matsubtract
try:
    from dislib.data.tensor import random_tensors, from_array, \
        from_pt_tensor, create_ds_tensor, from_ds_array  # noqa: F401
    imported_tensors = True
except Exception:
    print("WARNING: Tensors have not been loaded. No module named 'torch'.")
    imported_tensors = False
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
        print("WARNING: This dislib installation does not have a version "
              "number. "
              "Probably it was not installed with setup.py.\n%s" % e)
        __version__ = 'unknown'

if imported_tensors:
    __all__ = ['array', 'random_array', 'zeros', 'full', 'identity', 'eye',
               'load_txt_file', 'load_svmlight_file', 'load_npy_file',
               'load_mdcrd_file', 'matmul', 'matadd', 'matsubtract',
               'random_tensors', 'from_array', 'from_pt_tensor',
               'create_ds_tensor', 'from_ds_array',
               'save_txt', 'concat_rows', 'concat_columns',
               'apply_along_axis', 'kron', 'svd']
else:
    __all__ = ['array', 'random_array', 'zeros', 'full', 'identity', 'eye',
               'load_txt_file', 'load_svmlight_file', 'load_npy_file',
               'load_mdcrd_file', 'matmul', 'matadd', 'matsubtract',
               'save_txt', 'concat_rows', 'concat_columns',
               'apply_along_axis', 'kron', 'svd']

gpu_envar = os.environ.get('DISLIB_GPU_AVAILABLE', 'False')
__gpu_available__ = gpu_envar.lower() == 'true'

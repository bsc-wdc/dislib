import os

from dislib.data.array import random_array, apply_along_axis, array, \
    load_svmlight_file, load_txt_file

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
        print("Could not get installed dislib version. "
              "Probably it was not installed with setup.py.\n%s" % e)
        __version__ = 'unknown'

__all__ = ['load_txt_file', 'load_svmlight_file', 'random_array',
           'apply_along_axis', 'array']

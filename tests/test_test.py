import itertools
import uuid
from collections import defaultdict
from math import ceil

import numpy as np
import importlib
from pycompss.api.api import compss_wait_on

from pycompss.api.parameter import Type, COLLECTION_IN, Depth, COLLECTION_INOUT
from pycompss.api.task import task
from scipy import sparse as sp
from scipy.sparse import issparse, csr_matrix
from sklearn.utils import check_random_state

if importlib.util.find_spec("hecuba"):
    try:
        from hecuba.hnumpy import StorageNumpy
    except Exception:
        pass



bn, bm = (20, 5)
x = np.arange(100).reshape(10, -1)
data = StorageNumpy(input_array=x, name="test_array")
print("x: " + x)
print("data: " + data)
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

import gc
import os
import unittest

import numpy as np

os.environ["CONTACT_NAMES"] = "cassandra_container"
from hecuba import config
from pycompss.api.api import compss_wait_on
from sklearn.datasets import make_blobs

from pycompss.api.task import task    # Import @task decorator
from pycompss.api.parameter import *  # Import parameter metadata for the @task decorator

import dislib as ds
from dislib.cluster import KMeans
from dislib.decomposition import PCA
from dislib.neighbors import NearestNeighbors
from dislib.regression import LinearRegression
import time


config.session.execute("TRUNCATE TABLE hecuba.istorage")
config.session.execute("DROP KEYSPACE IF EXISTS hecuba_dislib")

x, y = make_blobs(n_samples=1500, random_state=170)
x_filtered = np.vstack(
    (x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))

block_size = (x_filtered.shape[0] // 10, x_filtered.shape[1])

x_train = ds.array(x_filtered, block_size=block_size)
x_train_hecuba = ds.array(x=x_filtered,
                          block_size=block_size)
x_train_hecuba.make_persistent(name="hecuba_dislib.test_array")

print(x_train)
l=StorageNumpy("hecuba_dislib.test_array")
while (l._numpy_full_loaded == false):
    x=1
print(x_train_hecuba._numpy_full_loaded)

#kmeans = KMeans(n_clusters=3, random_state=170)
#labels = kmeans.fit_predict(x_train).collect()

#kmeans2 = KMeans(n_clusters=3, random_state=170)
#h_labels = kmeans2.fit_predict(x_train_hecuba).collect()

#self.assertTrue(np.allclose(kmeans.centers, kmeans2.centers))
#self.assertTrue(np.allclose(labels, h_labels))


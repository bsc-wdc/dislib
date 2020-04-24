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
block_size = (2, 10)
x = np.array([[j for j in range(i * 10, i * 10 + 10)]
                      for i in range(10)])
data = ds.array(x=x, block_size=block_size)
print(data._blocks)
print(np.array(data._blocks).shape)

data.make_persistent(name="hecuba_dislib.test_array")

blocks = data._blocks
for block in blocks:
    del block
del data
gc.collect()

data=ds.load_from_hecuba(name="hecuba_dislib.test_array",block_size=block_size)
print(data._blocks)
print(np.array(data._blocks).shape)
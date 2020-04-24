import gc
import os
import unittest

import numpy as np

os.environ["CONTACT_NAMES"] = "cassandra_container"
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
from hecuba import config


def equal(arr1, arr2):
    equal = not (arr1 != arr2).any()

    if not equal:
        print("\nArr1: \n%s" % arr1)
        print("Arr2: \n%s" % arr2)

    return equal


@task(returns=1)
def test_already_persistent(x_train_hecuba):
    # import sys
    # sys.path.append("./debug/pydevd-pycharm.egg")
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('192.168.1.222', port=12345, stdoutToServer=True, stderrToServer=True)

    #copia = ds.load_from_hecuba(name="hecuba_dislib.test_array", block_size=block_size)
    import sys
    sys.path.append("./debug/pydevd-pycharm.egg")
    import pydevd_pycharm
    pydevd_pycharm.settrace('192.168.1.222', port=12345, stdoutToServer=True, stderrToServer=True)

    future=config.session.execute("TRUNCATE TABLE hecuba.istorage")
    # result = future.result()
    # trace = future.get_query_trace()
    # for e in trace.events:
    #     print(e.source_elapsed, e.description)
    config.session.execute_async("DROP KEYSPACE IF EXISTS hecuba_dislib", trace=True)
    x_train_hecuba.make_persistent(name="hecuba_dislib.test_array")
    return x_train_hecuba


def main():

    
    x, y = make_blobs(n_samples=1500, random_state=170)
    x_filtered = np.vstack(
            (x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))

    block_size = (x_filtered.shape[0] // 10, x_filtered.shape[1])
    print("shape del objeo")
    print(x_filtered.shape)

    x_train_hecuba = ds.array(x=x_filtered, block_size=block_size)
    
    # ensure that all data is released from memory
    # blocks = x_train_hecuba._blocks
    # for block in blocks:
    #     del block
    # del x_train_hecuba
    # gc.collect()
   
    value=test_already_persistent(x_train_hecuba)
    #copia = ds.load_from_hecuba(name="hecuba_dislib.test_array", block_size=block_size)
    value=compss_wait_on(value)
    print("FINAAAAL")
    print(value)
    


if __name__ == "__main__":
    main()
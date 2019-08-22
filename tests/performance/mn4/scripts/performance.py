import time

from pycompss.api.api import compss_barrier


def measure(name, dataset_name, model, x, y=None):
    print("==== STARTING ====", name)
    compss_barrier()
    s_time = time.time()
    model.fit(x, y)
    compss_barrier()
    print("==== OUTPUT ==== ", dataset_name, time.time() - s_time)

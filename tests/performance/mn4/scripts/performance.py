import time

from pycompss.api.api import compss_barrier


def measure(name, dataset_name, func, *args, **kwargs):
    print("==== STARTING ====", name, dataset_name)
    compss_barrier()
    s_time = time.time()
    func(*args, **kwargs)
    compss_barrier()
    print("==== TIME ==== ", name, dataset_name, time.time() - s_time)

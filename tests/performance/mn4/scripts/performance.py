import time

from pycompss.api.api import compss_barrier


def measure(name, model, dataset):
    print("==== STARTING ====", name)
    s_time = time.time()
    model.fit(dataset)
    compss_barrier()
    print("==== OUTPUT ==== ", time.time() - s_time)

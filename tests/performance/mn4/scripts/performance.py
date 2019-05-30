import time

from pycompss.api.api import compss_barrier


def measure(name, dataset_name, model, dataset):
    print("==== STARTING ====", name)
    s_time = time.time()
    model.fit(dataset)
    compss_barrier()
    print("==== OUTPUT ==== ", dataset_name, time.time() - s_time)

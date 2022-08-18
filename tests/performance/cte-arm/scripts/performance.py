import time
import os

from pycompss.api.api import compss_barrier, compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.task import task


def measure(name, dataset_name, func, *args, **kwargs):
    print("==== STARTING ====", name, dataset_name)
    compss_barrier()
    s_time = time.time()
    func(*args, **kwargs)
    compss_barrier()
    print("==== TIME ==== ", name, dataset_name, time.time() - s_time)
    print("In worker_working_dir: ", compss_wait_on(get_worker_working_dir()))


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def get_worker_working_dir():
    return os.getcwd()

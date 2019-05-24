#!/bin/bash -e

# Load COMPSs module. E.g.:

# export COMPSS_PYTHON_VERSION=3-ML
# module load COMPSs/2.5.pr

# Default process per worker
export ComputingUnits=48

# Run the tests/__main__.py file which calls all the tests named test_*.py
enqueue_compss \
    --num_nodes=2 \
    --qos=debug \
    --pythonpath=$(pwd) \
    --python_interpreter=python3 \
    --worker_working_dir=gpfs \
    ./tests/__main__.py &> >(tee tests_gpfs.log)

# Run the tests/__main__.py file which calls all the tests named test_*.py
enqueue_compss \
    --num_nodes=2 \
    --qos=debug \
    --pythonpath=$(pwd) \
    --python_interpreter=python3 \
    ./tests/__main__.py &> >(tee tests_scratch.log)


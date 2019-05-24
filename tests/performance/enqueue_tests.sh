#!/bin/bash -e

# Load desired COMPSs and dislib modules to be tested
module load dislib/unstable
module load COMPSs/2.5.pr


# Variables used for testing
als_dataset=/gpfs/projects/bsc19/COMPSs_DATASETS/dislib/recommendation/netflix/netflix_data_libsvm.txt
baseDir=$(pwd)


# ALS =========================================================================

# Sandbox
mkdir -p results/als
cd results/als

# The expected execution time of this test is ~441 seconds.
enqueue_compss \
    --qos=debug \
    --exec_time=30 \
    --num_nodes=5 \
    --worker_in_master_cpus=0 \
    --python_interpreter=python3 \
    ${baseDir}/als.py \
        --num_subsets=192 \
        --num_factors=100 \
        --data=${als_dataset}

cd ${baseDir}



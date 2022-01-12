#!/bin/bash -e
export COMPSS_PYTHON_VERSION=3
#workingDir=$(pwd)
workingDir=local_disk
numNodes=2
execTime=20
appdir=$PWD

enqueue_compss --qos=debug \
--worker_working_dir=$workingDir \
--agents \
--method_name="main" \
--num_nodes=$numNodes \
--log_level=debug \
--base_log_dir=/gpfs/scratch/bsc19/bsc19086/logs \
--exec_time=$execTime \
-d \
--lang=python \
--pythonpath=/gpfs/scratch/bsc19/bsc19086/dislib_nesting/tests/performance/mn4/scripts:/gpfs/scratch/bsc19/bsc19086/dislib_nesting/tests/performance/mn4/tests:/gpfs/scratch/bsc19/bsc19086/dislib_nesting/ \
example

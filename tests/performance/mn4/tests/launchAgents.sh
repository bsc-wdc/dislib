#!/bin/bash -e
export COMPSS_PYTHON_VERSION=3
workingDir=$(pwd)
#workingDir=local_disk
numNodes=2
execTime=60
appdir=$PWD
module load python/3.7.4


enqueue_compss --qos=debug \
--worker_working_dir=$workingDir \
--agents \
--method_name="main" \
--num_nodes=$numNodes \
--log_level=debug \
--base_log_dir=/gpfs/scratch/bsc19/bsc19086/logs \
--exec_time=$execTime \
-d \
--scheduler=es.bsc.compss.scheduler.fifodatanew.FIFODataScheduler \
-t \
--max_tasks_per_node=48 \
--lang=python \
--pythonpath=/gpfs/scratch/bsc19/bsc19086/dislib_nesting/tests/performance/mn4/scripts:/gpfs/scratch/bsc19/bsc19086/dislib_nesting/tests/performance/mn4/tests:/gpfs/scratch/bsc19/bsc19086/dislib_nesting/ \
example

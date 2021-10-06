file=${1:-"rf.py"}

enqueue_compss \
   -t \
   -d \
   --qos=debug \
   --python_worker_cache=true \
   --worker_in_master_cpus=0 \
   --worker_working_dir=/gpfs/scratch/bsc19/bsc19086/logs \
   --base_log_dir=/gpfs/scratch/bsc19/bsc19086/logs \
   --exec_time=20 \
   --max_tasks_per_node=48 \
   --num_nodes=2 \
   --scheduler=es.bsc.compss.scheduler.fifodatanew.FIFODataScheduler \
   --python_interpreter=python \
   --pythonpath=/gpfs/scratch/bsc19/bsc19086/dislib_cache/tests/performance/mn4/scripts:/gpfs/scratch/bsc19/bsc19086/dislib_cache/tests/performance/mn4/test:/gpfs/scratch/bsc19/bsc19086/dislib_cache/ \
   /gpfs/scratch/bsc19/bsc19086/dislib_cache/tests/performance/mn4/tests/${file}

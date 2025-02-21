#!/bin/bash -e
#
#SBATCH --job-name=postproc
#SBATCH -o /gpfs/projects/bsc19/PERFORMANCE/dislib/logs_FVN/post-%J.out
#SBATCH -e /gpfs/projects/bsc19/PERFORMANCE/dislib/logs_FVN/post-%J.err
#SBATCH --account=bsc19
#SBATCH --qos=gp_debug
#SBATCH -t 00:05:00


python /gpfs/projects/bsc19/PERFORMANCE/dislib/scripts/postprocess.py $1
python /gpfs/projects/bsc19/PERFORMANCE/dislib/scripts/plot.py $1


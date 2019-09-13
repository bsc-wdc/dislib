#!/bin/bash -e
#
#SBATCH --job-name=postproc
#SBATCH --workdir=.
#SBATCH -o /gpfs/projects/bsc19/PERFORMANCE/dislib/logs/post-%J.out
#SBATCH -e /gpfs/projects/bsc19/PERFORMANCE/dislib/logs/post-%J.err
#SBATCH -t 00:05:00


python /gpfs/projects/bsc19/PERFORMANCE/dislib/scripts/postprocess.py $1
python /gpfs/projects/bsc19/PERFORMANCE/dislib/scripts/plot.py $1


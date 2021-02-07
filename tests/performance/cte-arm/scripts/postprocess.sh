#!/bin/bash -e
#
#SBATCH --job-name=postproc
#SBATCH --workdir=.
#SBATCH -o /gpfs/projects/bsc19/PERFORMANCE/dislib/logs/post-%J.out
#SBATCH -e /gpfs/projects/bsc19/PERFORMANCE/dislib/logs/post-%J.err
#SBATCH -t 00:05:00

module purge && module load fuji/1.2.26b python/3.6.8
python3 /fefs/scratch/bsc19/bsc19029/PERFORMANCE/dislib/scripts/postprocess.py $time_str
python3 /fefs/scratch/bsc19/bsc19029/PERFORMANCE/dislib/scripts/plot.py $time_str


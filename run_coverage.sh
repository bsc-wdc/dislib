#!/bin/bash -e

# Run the coverage of the dislib using the tests in ./tests (sequential)
coverage3 run --data-file=cpu_cov --source dislib tests &
cpu_cov=$!

ssh compss@mt1.bsc.es mkdir -p /scratch/tmp/dislib-gpu-test
scp -r . compss@mt1.bsc.es:/scratch/tmp/dislib-gpu-test/
ssh compss@mt1.bsc.es cd /scratch/tmp/dislib-gpu-test;./run_gpu_cov.sh
scp compss@mt1.bsc.es:/scratch/tmp/dislib-gpu-test/gpu_cov .
ssh compss@mt1.bsc.es rm -rf /scratch/tmp/dislib-gpu-test

wait $cpu_cov

coverage3 combine cpu_cov gpu_cov

# Report coverage results to the CLI.
coverage3 report -m
# Upload coverage report to codecov.io
bash <(curl -s https://codecov.io/bash) -t 629589cf-e257-4262-8ec0-314dfd98f003

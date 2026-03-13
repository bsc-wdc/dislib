#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: launch.sh COMPSs_VERSION [DISLIB_VERSION]"
    exit 1
fi

export PRELOAD_PYTHON_LIBRARIES=/gpfs/projects/bsc19/PERFORMANCE/dislib/dislib/tests/performance/mn5/scripts/preimports.txt
module load hdf5
module load python/3.12.1
module load COMPSs/$1

if [ $# -ge 2 ]; then
    # If not provided, the scripts will add this dislib repo to the pythonpath when calling enqueue_compss
    module load dislib/$2
fi

export JAVA_HOME=/apps/GPP/COMPSs/JAVA/jdk-11.0.2/

python3 scripts/launch.py

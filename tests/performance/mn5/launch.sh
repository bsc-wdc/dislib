#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: launch.sh COMPSs_VERSION DISLIB_VERSION"
    exit 1
fi

export PRELOAD_PYTHON_LIBRARIES=/gpfs/projects/bsc19/PERFORMANCE/dislib/scripts/preimports.txt

module load hdf5
module load python/3.12.1
module load COMPSs/$1
module load dislib/$2
export JAVA_HOME=/apps/GPP/COMPSs/JAVA/jdk-11.0.2/

python3 scripts/launch.py

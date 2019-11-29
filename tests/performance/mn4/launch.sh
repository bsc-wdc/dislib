#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: launch.sh COMPSs_VERSION DISLIB_VERSION"
    exit 1
fi

module load dislib/$2
module load COMPSs/$1

python3 scripts/launch.py

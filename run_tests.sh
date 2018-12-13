#!/bin/bash


root_path="$(dirname "$(readlink -f "$0")")"

cd ${root_path}

# Default process per worker
export ComputingUnits=4

runcompss \
    --pythonpath=$(pwd) \
    --python_interpreter=python3 \
    ./tests/tests.py 2>&1 | tee output.log

if [ "${PIPESTATUS[0]}" == "1" ]; then 
    exit 1
fi

result=$(cat output.log | egrep "OK|FAILED")

echo "Tests result: ${result}"

if [[ $result =~ FAILED ]]; then 
    exit 1
fi

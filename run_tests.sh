#!/bin/bash -e


root_path="$(dirname "$(readlink -f "$0")")"

cd ${root_path}
mkdir -p tests/log
rm -f tests/log/*

# Default process per worker
export ComputingUnits=4


for test_path in $(ls ./tests/*py); do
    test_file=$(basename -- "${test_path}")
    # We don't want to run the __init__.py file
    if [ "${test_file}" == "__init__.py" ]; then
        continue
    fi
    runcompss \
        --pythonpath=$(pwd) \
        --python_interpreter=python3 \
        ./tests/${test_file} &> >(tee ./tests/log/${test_file}.log)

    # (1) Script fails fast (-e) so at this point testing will stop if runcompss returns 1

    # Check the unittest result
    result=$(cat tests/log/${test_file}.log | egrep "OK|FAILED")

    # Print the unittest result
    echo "[${test_file}] Test result: ${result}" > >(tee -a tests/log/results.log)

    # If the unittest fail the testing is stopped to mirror the (-e) behaviour in (1)
    if [[ $result =~ FAILED ]]; then 
        exit 1
    fi
done

cat tests/log/results.log

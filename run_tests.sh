#!/bin/bash -e

# Default process per worker
#export ComputingUnits=4
echo "Using Cassandra host $CONTACT_NAMES"
#echo "export CONTACT_NAMES=$CONTACT_NAMES" >> ~/.bashrc

# Run the tests/__main__.py file which calls all the tests named test_*.py
runcompss \
     --pythonpath="/usr/local/lib/python3.6/dist-packages/Hecuba-0.1.3.post1-py3.6-linux-x86_64.egg/" \
     --python_interpreter=python3 \
     --classpath=/hecuba_repo/storageAPI/storageItf/target/StorageItf-1.0-jar-with-dependencies.jar \
     --storage_conf="/dislib/storage_conf.cfg" \
     /dislib/tests/test_hecuba.py &> >(tee output.log)

# Check the unittest output because PyCOMPSs exits with code 0 even if there
# are failed tests (the execution itself is successful)
result=$(cat output.log | egrep "OK|FAILED")

echo "Tests result: ${result}"

# If word Failed is in the results, exit 1 so the pull request fails
if [[ $result =~ FAILED ]]; then 
        exit 1
    fi


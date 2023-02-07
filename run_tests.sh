#!/bin/bash -e

# Default process per worker
export ComputingUnits=4

# Run the tests/__main__.py file which calls all the tests named test_*.py
runcompss \
    --pythonpath=$(pwd) \
    --python_interpreter=python3 \
    ./tests/__main__.py &> >(tee output.log)
    
    
echo "Checking for error..."

if grep -q "TOTALLY FAILED" output.log; then
   job_id=$(grep -oP "\d+(?=\|)" output.log)
   job_file="/root/.COMPSs/__main__.py_01/jobs/job${job_id}_NEW.err"
   echo $job_file
   cat $job_file
fi

# Check the unittest output because PyCOMPSs exits with code 0 even if there
# are failed tests (the execution itself is successful)
result=$(cat output.log | egrep "OK|FAILED")

echo "Tests result: ${result}"

# If word Failed is in the results, exit 1 so the pull request fails
if [[ $result =~ FAILED ]]; then 
        exit 1
fi

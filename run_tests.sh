#!/bin/bash -e

# Declare paths
COMPSS_HOME="/opt/COMPSs"
resources_file="${COMPSS_HOME}/Runtime/configuration/xml/resources/default_resources.xml"

# Default process per worker
export ComputingUnits=4
export ComputingUnitsGPUs=0

# Total computing units used by COMPSs
TotalComputingUnits=8

# Increase available number of computing units
sed -i "s|<ComputingUnits>[^<]*</ComputingUnits>|<ComputingUnits>$TotalComputingUnits</ComputingUnits>|" $resources_file
ActualComputingUnits=$(sed -n 's|.*<ComputingUnits>\([^<]*\)</ComputingUnits>.*|\1|p' $resources_file)
echo "Using a total of $ActualComputingUnits computing units"

# Run the tests/__main__.py file which calls all the tests named test_*.py
runcompss \
    --pythonpath=$(pwd) \
    --python_interpreter=python3 \
    ./tests/__main__.py &> >(tee output.log)

# Check the unittest output because PyCOMPSs exits with code 0 even if there
# are failed tests (the execution itself is successful)
result=$(cat output.log | egrep "OK|FAILED")

echo "Tests result: ${result}"

# If word Failed is in the results, exit 1 so the pull request fails
if [[ $result =~ FAILED ]]; then 
        exit 1
fi

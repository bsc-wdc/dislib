#!/bin/bash -e

# Test, coverage, and style should be run from the repo's root
export root_path="$(dirname "$(readlink -f "$0")")"
cd ${root_path}

# Add dislib to the python path
export PYTHONPATH=$PYTHONPATH:${root_path}

echo "Running flake8 style check"
./run_style.sh

echo "Running tests"
# Run the tests in ./tests with PyCOMPSs
./run_tests.sh

echo "Running code coverage"
./run_coverage.sh

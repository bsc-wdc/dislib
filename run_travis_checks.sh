#!/bin/bash -e

root_path="$(dirname "$(readlink -f "$0")")"
cd ${root_path}

echo "Running flake8 style check"
./run_style.sh

echo "Running tests"
./run_tests.sh

echo "Running code coverage"
./run_coverage.sh

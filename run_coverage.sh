#!/bin/bash -e

# Run the coverage of the dislib using the tests in ./tests (sequential)
coverage3 run --source dislib tests
# Report coverage results to the CLI.
coverage3 report -m
# Upload coverage report to codecov.io
bash <(curl -s https://codecov.io/bash)
#!/bin/bash -e

# Run the coverage of the dislib using the tests in ./tests (sequential)
coverage run --source dislib tests
coverage run -a --source dislib tests_nesting
# Create the report
coverage report
# Report coverage results to the CLI.
coverage report -m
# Upload coverage report to codecov.io
bash <(curl -s https://codecov.io/bash) -t 629589cf-e257-4262-8ec0-314dfd98f003

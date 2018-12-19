#!/bin/bash -e

# Runs flake8 code style checks on the dislib. The command output should be
# empty which indicates that no style issues were found.
python3 -m flake8 --exclude=docs/scipy-sphinx-theme .

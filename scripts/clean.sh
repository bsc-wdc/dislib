#!/bin/bash -e

##############################
# Clean any unnecesary files #
##############################

# Go to root folder
cd ..

# Clean
rm -rf dist
rm -rf dislib.egg-info
find | grep __pycache__ | xargs rm -rf
find . -name "*.pyc" -exec rm -f {} \;

# Go back to scripts folder
cd scripts
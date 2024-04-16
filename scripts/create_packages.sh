#!/bin/bash -e

##############################
#       Build packages       #
##############################

# Go to root folder
cd ..

# Build
python3 -m build .

# Go back to scripts folder
cd scripts
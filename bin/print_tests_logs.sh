#!/bin/bash


# Iterates over all the files found in .COMPSs (in testing only __main__.py folder should contain data).
# For each file prints the file name and its contents.

for f in $(find /root/.COMPSs/ -follow); do 
    if [[ -f $f ]]; then  # only for files
        printf "\n==> $f <==\n"
        cat $f
    fi
done

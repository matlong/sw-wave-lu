#!/bin/bash

#echo "Bash version ${BASH_VERSION}..."

path_src=Desktop

for i in $(seq -f "%03g" 1 18)
do
    echo "$path_src/$i"
done


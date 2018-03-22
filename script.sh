#!/bin/bash
#echo "testing data..."
for ((x=1;x<=5;x++)); do cat iris-data.txt | ./split.sh 10 ./id3 $x ; done
#echo "done."
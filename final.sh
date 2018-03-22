#!/bin/bash
./script.sh > right
awk '{ total += $1 } END { print total/NR }' right
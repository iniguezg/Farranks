#!/usr/local/bin/bash

pnu=0

for ptau in $(seq 0 0.02 1)
do
	echo "ptau = $ptau"

	python script_fitData_01.py $1 $ptau $pnu $2
done

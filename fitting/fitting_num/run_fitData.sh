#!/usr/local/bin/bash

for ptau in $(seq 0 0.02 1)
do
	echo "ptau = $ptau"

	for pnu in $(seq 0 0.02 1)
	do
		echo "pnu = $pnu"

		python script_fitData_01.py $1 $ptau $pnu $2
	done
done

#!/bin/bash
#SBATCH --time=05-00
#SBATCH --mem-per-cpu=2G

srun python script_fitData.py $1

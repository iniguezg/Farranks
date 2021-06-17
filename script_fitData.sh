#!/bin/bash
#SBATCH --time=05-00

srun python script_fitData.py $1

# Farranks

## Farranks - Exploring rank dynamics in complex systems

This library includes code to build, analyze, and visualize empirical datasets and model simulations of rank dynamics in complex systems. The description and results of ranking data and models can be found in the following publication. If you use this code, please cite it as:

G. Iñiguez, C. Pineda, C. Gershenson, A.-L. Barabási  
Dynamics of ranking  
To appear in *Nature Communications* (2022)  
[arXiv: 2104.13439](https://arxiv.org/abs/2104.13439)


### CONTENTS

#### Main folder

Core library:
- **data_misc.py** (functions to process and analyze empirical rank data)
- **props_misc.py** (functions to compute rank properties of data/model)
- **model_misc.py** (functions to run and analyze model of rank dynamics)

Sample analysis scripts:
- **script_fitData.py** (script for fitting data with model)
- **script_getData.py** (script for getting data properties)
- **script_solveFitEqs.py** (script for solving and visualizing fit equations)

Secondary scripts:
- **script_fitData.sh** (script to run code in cluster)
- **script_fitData.slurm** (script to run code in cluster)
- **script_moveFiles.py** (script to move files in bulk)

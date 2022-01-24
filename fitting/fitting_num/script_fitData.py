#! /usr/bin/env python

#Farranks - Exploring rank dynamics in complex systems
#Copyright (C) 2022 Gerardo Iñiguez

### SCRIPT FOR RUNNING NULL MODEL IN FARRANKS PROJECT ###

#import modules
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import expanduser

import model_misc


## RUNNING MODEL SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#input arguments
	dataname = sys.argv[1] #dataset to explore
	ptau, pnu = float( sys.argv[2] ), float( sys.argv[3] ) #parameters to explore
	ntimes = int( sys.argv[4] ) #number of model realisations

	#flags and locations
	loadflag = 'n'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_data = root_loc+'nullModel/v4/files/' #location of output files
	saveloc_model = root_loc+'nullModel/v4/files/model/'

	## DATA ##

	#get parameters for all datasets and selected dataset
	params_data = pd.read_pickle( saveloc_data+'params_data.pkl' )
	params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)

	## MODEL ##

	#set model parameters
	params[ 'ptau' ], params[ 'pnu' ] = ptau, pnu #set parameters for model
	params[ 'ntimes' ] = ntimes

	#run/load model
	prop_dict = model_misc.model_props( [ 'flux', 'rank', 'open', 'disp', 'succ' ], params, loadflag=loadflag, saveloc=saveloc_model, saveflag='y' )

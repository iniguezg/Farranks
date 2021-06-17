#! /usr/bin/env python

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


#DEBUGGIN'
	# #set model parameters
	# params = { 'N' : 2614, 'N0' : 1041, 'T' : 45, 'ptau' : 0.1, 'pnu' : 0.01, 'ntimes' : 5 }
	# N0, T = params['N0'], params['T'] #get parameters from data

	# #openness
	# prop_dict = model_misc.model_props( ['open'], params, loadflag, saveloc )
	# openprops_model, = prop_dict['open']
	# openness_model = openprops_model.loc[ 'openness' ]
	#
	# xplot = openness_model.index / float( T ) #normalised time = ( 0, ..., T-1 ) / T
	# yplot_model = openness_model
	# plt.clf()
	# plt.semilogy( xplot, yplot_model )
	# plt.axis([ -0.05, 1.05, 1, 100 ])
	# plt.show()

	# #flux
	# prop_dict = model_misc.model_props( ['flux'], params, loadflag, saveloc )
	# fluOprops_model, fluIprops_model = prop_dict['flux']
	# flux_model = fluIprops_model.mean( axis=1 ) #IN-flow (=OUT-flow when averaged over Nt)
	#
	# xplot = np.arange( 1, T ) / T #normalised time = ( 1, ..., T-1 ) / T
	# yplot_model = flux_model
	# plt.clf()
	# plt.plot( xplot, yplot_model )
	# plt.axis([ -0.05, 1.05, -0.05, 1.05 ])
	# plt.show()

	# #change
	# prop_dict = model_misc.model_props( ['rank'], params, loadflag, saveloc )
	# rankprops_model, = prop_dict['rank']
	# rankchange_model = rankprops_model.loc[ 'rankchange' ]
	# xplot = np.arange( 1, N0+1 ) / N0 #normalised rank = ( 1, ..., N0 ) / N0
	# yplot_model = rankchange_model
	# plt.clf()
	# plt.plot( xplot, yplot_model )
	# plt.axis([ -0.05, 1.05, -0.05, 1.05 ])
	# plt.show()

	### analysis 1: fit null model with property from dataset ##

	#model_fit = data_misc.data_estimate_params( prop_name, dataname, params, saveloc )
	#
	#print( model_fit.fit_report() )

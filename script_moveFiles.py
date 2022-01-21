#! /usr/bin/env python

#Farranks - Exploring rank dynamics in complex systems
#Copyright (C) 2022 Gerardo Iñiguez

### SCRIPT FOR MOVING MISC FILES IN FARRANKS PROJECT ###

#import modules
import sys
import pandas as pd
import subprocess as sp
from os.path import expanduser

import data_misc


## RUNNING DATA SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#model fit parameters
	ntimes = 10 #number of model realisations
	prop_names = { 'flux_time' : [ 'fluIprops_', 'fluOprops_' ],
				   'openness' : [ 'openprops_' ],
				   'flux_rank' : [ 'fluIprops_', 'fluOprops_' ],
				   'change' : [ 'rankprops_' ],
				   'diversity' : [ 'rankprops_' ],
				   'success' : [ 'success_', 'surprise_' ] } #properties to fit

	#flag and locations
	loadflag = 'y'
	location = expanduser('~') + '/prg/xocial/Farranks/diversity/Database/' #root location of datasets
#	saveloc_data = 'files/' #location of output files
#	saveloc_model = 'files/model/'
#	saveloc_temp = 'files/model/run_01/'
	saveloc_data = '/storage/bases_redes/iniguez/prg/xocial/Farranks/nullModel/v4/files/'
	saveloc_model = '/storage/bases_redes/iniguez/prg/xocial/Farranks/nullModel/v4/files/model/send/'
	saveloc_temp = '/storage/bases_redes/iniguez/prg/xocial/Farranks/nullModel/v4/files/model/run_01/'

#	#datasets to explore
#	datasets = [ 'AcademicRanking' ]
#	datasets = [ 'AtlasComplex', 'Football_FIFA', 'Nascar_WinstonCupGrandNational', 'Hienas', 'Nascar_BuschGrandNational', 'Cities_RU', 'Fortune', 'Football_Scorers' ]
	datasets = [ 'Citations', 'english', 'enron-sent-mails-weekly', 'FIDEFemale', 'FIDEMale', 'french', 'german', 'github-watch-weekly', 'Golf_OWGR', 'italian', 'Poker_GPI', 'russian', 'spanish', 'Tennis_ATP', 'TheGuardian_avgRecommends', 'TheGuardian_numberComments' ]
#	datasets = [ 'Cities_UK', 'Earthquakes_avgMagnitude', 'Earthquakes_numberQuakes', 'metroMex', 'UndergroundByWeek' ]


	## DATA AND MODEL ##

	#get parameters for all datasets
	params_data = pd.read_pickle( saveloc_data+'params_data.pkl' )

	for dataname in datasets: #loop through considered datasets
		print( 'dataset name: ' + dataname ) #print dataset

		#get parameters for dataset
		params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data

		#get model parameters for selected dataset
		params[ 'ntimes' ] = ntimes
		params_model = data_misc.data_estimate_params( dataname, params, loadflag, saveloc_data, saveloc_model )

		## MOVE FILES ##

		for prop_name, prop_prefixes in prop_names.items(): #loop through measures to fit

			#set parameters per rank property
			params['pnu'], params['ptau'] = params_model.loc[ prop_name, [ 'pnu', 'ptau' ] ]

			#filename for output files
			param_str_model = 'model_N{}_N0{}_T{}_ptau{:.2f}_pnu{:.2f}_ntimes{}.pkl'.format( params['N'], params['N0'], params['T'], params['ptau'], params['pnu'], ntimes )

			#move files
			for prop_prefix in prop_prefixes: #loop through files to copy
				sp.run( [ 'cp', '-pf',
					  	saveloc_temp + prop_prefix + param_str_model,
					  	saveloc_model ] )

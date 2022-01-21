#! /usr/bin/env python

#Farranks - Exploring rank dynamics in complex systems
#Copyright (C) 2022 Gerardo Iñiguez

### SCRIPT FOR GETTING DATA PROPERTIES IN FARRANKS PROJECT ###

#import modules
import sys
import pandas as pd
from os.path import expanduser

import data_misc#, model_misc


## RUNNING DATA SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#flags to run analyses independently
	flag1, flag2, flag3, flag4, flag5, flag6, flag7, flag8, flag9 = False, False, False, False, False, False, False, False, True

	thres = 0.5 #transition threshold

	#input arguments
	loadflag = sys.argv[1] #load flag ( 'y', 'n' )

	location = expanduser('~') + '/prg/xocial/Farranks/diversity/Database/' #root location of datasets
	saveloc = 'files/' #location of output files

	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_data = root_loc+'nullModel/v4/files/' #location of output files

	saveloc_model = 'files/model/run_01/'
#	saveloc = '/storage/bases_redes/iniguez/prg/xocial/Farranks/nullModel/v4/files/'
#	saveloc_model = '/storage/bases_redes/iniguez/prg/xocial/Farranks/nullModel/v4/files/model/'


	## analysis 1: get parameters for all datasets ##
	if flag1:
		#datasets to explore
		datasets = [ 'AcademicRanking', 'AtlasComplex', 'Citations', 'Cities_RU', 'Cities_UK', 'Earthquakes_avgMagnitude', 'Earthquakes_numberQuakes', 'english', 'enron-sent-mails-weekly', 'FIDEFemale', 'FIDEMale', 'Football_FIFA', 'Football_Scorers', 'Fortune', 'french', 'german', 'github-watch-weekly', 'Golf_OWGR', 'Hienas', 'italian', 'metroMex', 'Nascar_BuschGrandNational', 'Nascar_WinstonCupGrandNational', 'Poker_GPI', 'russian', 'spanish', 'Tennis_ATP', 'TheGuardian_avgRecommends', 'TheGuardian_numberComments', 'UndergroundByWeek' ]

		params_data = data_misc.data_params( datasets, location, loadflag, saveloc )


	## analysis 2: get time intervals for all datasets ##
	if flag2:
		#formats for time intervals
		formats = { 'AcademicRanking' : '%Y', 'AtlasComplex' : '%Y', 'Citations' : '%Y', 'Cities_RU' : '%Y', 'Cities_UK' : '%Y', 'Earthquakes_avgMagnitude' : '%Y-%m-%d', 'Earthquakes_numberQuakes' : '%Y-%m-%d', 'english' : '%Y', 'enron-sent-mails-weekly' : '%d.%m.%Y', 'FIDEFemale' : '%b%y', 'FIDEMale' : '%b%y', 'Football_FIFA' : '%b%Y', 'Football_Scorers' : 'wk%W-%Y', 'Fortune' : '%Y', 'french' : '%Y', 'german' : '%Y', 'github-watch-weekly' : '%d.%m.%Y', 'Golf_OWGR' : '%d-%m-%Y', 'Hienas' : '%Y', 'italian' : '%Y', 'metroMex' : '%d-%b', 'Nascar_BuschGrandNational' : '%Y', 'Nascar_WinstonCupGrandNational' : '%Y', 'Poker_GPI' : '%Y-%m-%d', 'russian' : '%Y', 'spanish' : '%Y', 'Tennis_ATP' : '%d.%m.%Y', 'TheGuardian_avgRecommends' : '%b-%d-%Y', 'TheGuardian_numberComments' : '%b-%d-%Y', 'UndergroundByWeek' : 'day_%d%m%y_hr_%H_%M', 'VideogameEarnings' : '%Y', 'Virus' : 'day_%j' }

		intervals_data = data_misc.data_intervals( datasets, formats, location, loadflag, saveloc )


	## analysis 3: get rank/element time series for all datasets ##
	if flag3:
		#get parameters for all datasets (assume loadflag = y)
		params_data = data_misc.data_params( datasets, location, 'y', saveloc )

		for dataname in datasets: #loop through considered datasets
			print( 'dataset name: ' + dataname ) #print dataset
			params = params_data.loc[ dataname ] #get parameters for dataset

			rankseries, elemseries = data_misc.data_series( dataname, params, location, loadflag, saveloc )


	## analysis 4: get score time series for all datasets ##
	if flag4:
		#get parameters for all datasets (assume loadflag = y)
		params_data = data_misc.data_params( datasets, location, 'y', saveloc )

		for dataname in datasets: #loop through considered datasets
			print( 'dataset name: ' + dataname ) #print dataset
			params = params_data.loc[ dataname ] #get parameters for dataset

			scorseries = data_misc.data_scores( dataname, params, location, loadflag, saveloc )


	## analysis 5: get flux/rank/open/succ/mle/tran properties for rank/element time series of all datasets ##
	if flag5:
		datasets = [ 'AcademicRanking', 'AtlasComplex', 'Citations', 'Cities_RU', 'Cities_UK', 'Earthquakes_avgMagnitude', 'Earthquakes_numberQuakes', 'enron-sent-mails-weekly', 'Football_FIFA', 'Football_Scorers', 'Fortune', 'Hienas', 'metroMex', 'Nascar_BuschGrandNational', 'Nascar_WinstonCupGrandNational', 'UndergroundByWeek' ]
		# datasets = [ 'english', 'FIDEFemale', 'FIDEMale', 'french', 'german', 'github-watch-weekly', 'Golf_OWGR', 'italian', 'Poker_GPI', 'russian', 'spanish', 'Tennis_ATP', 'TheGuardian_avgRecommends', 'TheGuardian_numberComments' ]
		# datasets = [ 'github-watch-weekly' ]

		#get parameters for all datasets (assume loadflag = y)
		params_data = data_misc.data_params( datasets, location, 'y', saveloc )

		for dataname in datasets: #loop through considered datasets
			print( 'dataset name: ' + dataname ) #print dataset
			params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)

			params['thres'] = thres #transition threshold

			fluOprops, fluIprops = data_misc.data_props( 'flux', dataname, params, location, loadflag, saveloc )

			rankprops = data_misc.data_props( 'rank', dataname, params, location, loadflag, saveloc )

			openprops = data_misc.data_props( 'open', dataname, params, location, loadflag, saveloc )

			dispprops = data_misc.data_props( 'disp', dataname, params, location, loadflag, saveloc )

			success, surprise = data_misc.data_props( 'succ', dataname, params, location, loadflag, saveloc )

			mleTprops, mleDprops = data_misc.data_props( 'mle', dataname, params, location, loadflag, saveloc )

			tranprops = data_misc.data_props( 'tran', dataname, params, location, loadflag, saveloc )


	## analysis 6: get mean flux and mean openness deriv of all datasets ##
	if flag6:
		#get parameters for all datasets (assume loadflag = y)
		params_data = data_misc.data_params( datasets, location, 'y', saveloc )

		print( 'mean flux\tmean openness deriv\tdataset' )

		for dataname in datasets: #loop through considered datasets
			params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)
			N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
			param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

			#mean flux over time/ranks
			fluOprops_data = pd.read_pickle( saveloc_data + 'fluOprops_' + param_str_data )
			#average OUT-/IN- flux over time, and then over ranks
			flux_data = fluOprops_data.mean( axis=0 ).mean()

			#average openness derivstive
			openprops_data = pd.read_pickle( saveloc_data + 'openprops_' + param_str_data )
			open_deriv_data = openprops_data.loc[ 'open_deriv' ].mean() #get mean

			print( '{:.3f}\t\t{:.3f}\t\t\t{}'.format( flux_data, open_deriv_data, dataname ) )


	## analysis 7: get model phase diagram and optimal parameters of all datasets ##
	if flag7:
		datasets = [ 'AtlasComplex', 'Citations', 'Cities_RU', 'english', 'enron-sent-mails-weekly', 'FIDEFemale', 'FIDEMale', 'Football_FIFA', 'Football_Scorers', 'Fortune', 'french', 'german', 'github-watch-weekly', 'Golf_OWGR', 'Hienas', 'italian', 'Nascar_BuschGrandNational', 'Nascar_WinstonCupGrandNational', 'Poker_GPI', 'russian', 'spanish', 'Tennis_ATP', 'TheGuardian_avgRecommends', 'TheGuardian_numberComments' ]
		ntimes = 10 #number of model realisaations

		#get parameters for all datasets (assume loadflag = y)
		params_data = data_misc.data_params( datasets, location, 'y', saveloc )

		for dataname in datasets: #loop through considered datasets
			print( 'dataset name: ' + dataname ) #print dataset
			params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)
			params[ 'ntimes' ] = ntimes #set number of realisations

	#		#get model phase diagram for given dataset
	#		model_PD = data_misc.data_estimate_PD( dataname, params, loadflag, saveloc, saveloc_model )

			#get model optimal parameters for given dataset
			#NOTE: (local) optimal parameters ONLY along ptau = pnu diagonal!
			params_model = data_misc.data_estimate_params( dataname, params, loadflag, saveloc, saveloc_model )


	## analysis 8: (sub)sample rank/element time series / parameters of all datasets
	if flag8:
		location = 'files/' #original rank/element time series files
		saveloc = 'files/sampling/' #sampling output files
		# datasets = [ 'AcademicRanking' ]

		#get parameters for all datasets
		params_data = pd.read_pickle( location+'params_data.pkl' )
		for dataname in datasets: #loop through considered datasets
			print( 'dataset name: ' + dataname ) #print dataset
			params = params_data.loc[ dataname ] #get parameters for dataset

			#(sub)sample rank/element time series / parameters for given dataset
			params_sample = data_misc.data_params_sample( dataname, params, location, loadflag, saveloc )


	## analysis 9: get flux/open/succ properties for sample rank/element time series of all datasets ##
	if flag9:
		location = 'files/' #original rank/element time series files
		saveloc = 'files/sampling/' #sampling output files
		datasets = [ 'AcademicRanking' ]

		#get parameters for all datasets
		params_data = pd.read_pickle( location+'params_data.pkl' )
		for dataname in datasets: #loop through considered datasets
			print( 'dataset name: ' + dataname ) #print dataset

			params = params_data.loc[ dataname ] #get parameters for original/sample datasets
			params_sample = pd.read_pickle( saveloc+'params_sample_{}.pkl'.format( dataname ) )

			for jump in range( 2, params['T'] ): #loop through (all possible) jump values
				print( '\tjump: {}'.format(jump) ) #print jump
				dataname_sample = dataname + '_jump{}'.format( jump ) #pick sample
				params_sample_jump = params_sample.loc[ jump ] #and get its parameters

				fluOprops, fluIprops = data_misc.data_props( 'flux', dataname_sample, params_sample_jump, location, loadflag, saveloc )

				openprops = data_misc.data_props( 'open', dataname_sample, params_sample_jump, location, loadflag, saveloc )

				success, surprise = data_misc.data_props( 'succ', dataname_sample, params_sample_jump, location, loadflag, saveloc )

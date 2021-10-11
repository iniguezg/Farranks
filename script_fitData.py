#! /usr/bin/env python

### SCRIPT FOR FITTING NULL MODEL WITH DATA IN FARRANKS PROJECT ###

#import modules
import sys
import pandas as pd
from os.path import expanduser

import data_misc, model_misc


## RUNNING MODEL SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#fitting variables
#	par_step = 0.001 #parameter step
#	gof_str = 'MSE' #Goodness-of-fit measure. Default: Mean Squared Error (MSE)

	#flags and locations
	loadflag = 'n'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	# saveloc_data = root_loc+'nullModel/v4/files/' #location of output files
	# saveloc_mle = root_loc+'nullModel/v4/files/params/params_mle/'
	saveloc_data = '/m/cs/scratch/networks/inigueg1/prg/xocial/Farranks/nullModel/v4/files/'

	#datasets to explore
	datasets = [ 'AcademicRanking', 'AtlasComplex', 'Citations', 'Cities_RU', 'Cities_UK', 'Earthquakes_avgMagnitude', 'Earthquakes_numberQuakes', 'english', 'enron-sent-mails-weekly', 'FIDEFemale', 'FIDEMale', 'Football_FIFA', 'Football_Scorers', 'Fortune', 'french', 'german', 'github-watch-weekly', 'Golf_OWGR', 'Hienas', 'italian', 'metroMex', 'Nascar_BuschGrandNational', 'Nascar_WinstonCupGrandNational', 'Poker_GPI', 'russian', 'spanish', 'Tennis_ATP', 'TheGuardian_avgRecommends', 'TheGuardian_numberComments', 'UndergroundByWeek' ]
#	datasets = [ 'FIDEFemale', 'FIDEMale', 'Golf_OWGR', 'Poker_GPI', 'Tennis_ATP', 'TheGuardian_avgRecommends', 'TheGuardian_numberComments', 'english', 'french', 'german', 'github-watch-weekly', 'italian', 'russian', 'spanish' ]
#	datasets = [ 'enron-sent-mails-weekly', 'Football_FIFA', 'Football_Scorers', 'Fortune', 'Hienas', 'metroMex', 'Nascar_BuschGrandNational', 'Nascar_WinstonCupGrandNational', 'UndergroundByWeek' ]
#	datasets = [ 'VideogameEarnings', 'Virus' ] #shady data

	datatypes = { 'AcademicRanking' : 'open', 'AtlasComplex' : 'open', 'Citations' : 'open', 'Cities_RU' : 'open', 'Cities_UK' : 'closed', 'Earthquakes_avgMagnitude' : 'closed', 'Earthquakes_numberQuakes' : 'closed', 'english' : 'open', 'enron-sent-mails-weekly' : 'open', 'FIDEFemale' : 'open', 'FIDEMale' : 'open', 'Football_FIFA' : 'closed', 'Football_Scorers' : 'open', 'Fortune' : 'open', 'french' : 'open', 'german' : 'open', 'github-watch-weekly' : 'open', 'Golf_OWGR' : 'open', 'Hienas' : 'open', 'italian' : 'open', 'metroMex' : 'closed', 'Nascar_BuschGrandNational' : 'open', 'Nascar_WinstonCupGrandNational' : 'open', 'Poker_GPI' : 'open', 'russian' : 'open', 'spanish' : 'open','Tennis_ATP' : 'open', 'TheGuardian_avgRecommends' : 'open', 'TheGuardian_numberComments' : 'open', 'UndergroundByWeek' : 'closed' } #type dict
#	datasets = { 'VideogameEarnings' : 'open', 'Virus' : 'open' } #shady data

	# for dataname in datasets: #loop through considered datasets
	# 	print( 'dataset name: ' + dataname ) #print dataset
	#
	# 	## DATA ##
	#
	# 	#get parameters for all datasets and selected dataset
	# 	params_data = pd.read_pickle( saveloc_data+'params_data.pkl' )
	# 	params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)
	#
	# 	#model fit parameters
	# 	datatype = datatypes[ dataname ] #dataset type: open, closed

		# ## analysis 1: get model PD for dataset ##
		#
		# #get model phase diagram for given dataset (theo)
		# model_PD = data_misc.data_estimate_PD_theo( dataname, params, loadflag, saveloc_data, par_step=par_step, gof_str=gof_str, datatype=datatype )
		#
		#
		# ## analysis 2: get optimal parameters for dataset ##
		#
		# params_model = data_misc.data_estimate_params_theo( dataname, params, loadflag, saveloc_data, gof_str=gof_str, datatype=datatype )


		# ## analysis 3: get optimal parameters for dataset (all parameters at same time) ##
		#
		# params_model = data_misc.data_estimate_params_all( dataname, params, loadflag, saveloc_data, datatype=datatype )
		# print( params_model )


		# ## analysis 4: get optimal parameters for dataset (maximum likelihood estimation) ##
		#
		# #do simple fitting first
		# params_model = data_misc.data_estimate_params_all( dataname, params, 'y', saveloc_data, datatype=datatype )
		# #do MLE fitting
		# params_model_mle = data_misc.data_estimate_params_MLE( dataname, params, loadflag, saveloc_data, datatype=datatype, sample_frac=0.1 )
		# #print status
		# print( '\tBASIC: pnu = {},\tptau = {}\n\tMLE: pnu = {},\tptau = {}\n'.format( params_model.loc['optimal', 'pnu'], params_model.loc['optimal', 'ptau'], params_model_mle.loc['optimal', 'pnu'], params_model_mle.loc['optimal', 'ptau'] ) )


		# ## analysis 5: get optimal parameters for samples of dataset (all parameters at same time)
		#
		# location = 'files/' #original rank/element time series files
		# saveloc = 'files/sampling/' #sampling output files
		#
		# params_model = data_misc.data_estimate_params_sample( dataname, params, location, loadflag, saveloc, datatype=datatype )
		# print(params_model)


	## analysis 6: estimate system size in model that leads to number of elements ever in ranking in data

	# ntimes = 2500 #number of realisations (for bootstrapping)
	ntimes = 100
	dataname = sys.argv[1] #considered dataset
	print( 'dataset name: ' + dataname ) #print dataset

	#get parameters for all datasets and selected dataset
	params_data = pd.read_pickle( saveloc_data+'params_data.pkl' )
	params = params_data.loc[ dataname ]
	params['ntimes'] = ntimes

	params_size = model_misc.estimate_params_size( dataname, params, loadflag, saveloc_data )
	print( 'N_est = {}'.format(params_size), flush=True )


	# ## analysis 7: estimate parameter deviations for dataset (all parameters at same time)
	#
	# ntimes = 2500 #number of realisations (for bootstrap sampling)
	# dataname = sys.argv[1] #considered dataset
	# print( 'dataset name: ' + dataname ) #print dataset
	#
	# #get parameters for all datasets and selected dataset
	# params_data = pd.read_pickle( saveloc_data+'params_data.pkl' )
	# params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)
	# params['ntimes'] = ntimes
	# datatype = datatypes[ dataname ] #dataset type: open, closed
	#
	# params_devs = data_misc.data_estimate_params_devs( dataname, params, loadflag, saveloc_data, datatype=datatype )
	# print( params_devs, flush=True )


#DEBUGGIN'

		# N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		# p0 = N0 / float( N ) #ranking fraction
		# param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )
		#
		# MLEprops_data = pd.read_pickle( saveloc_data + 'MLEprops_' + param_str_data )
		# surv_time_mean = MLEprops_data.loc['survival'].mean()
		#
		# openprops_data = pd.read_pickle( saveloc_data + 'openprops_' + param_str_data )
		# open_deriv_data = openprops_data.loc[ 'open_deriv' ].mean()
		#
		# print( 'open_deriv = {}, (2-p)/<t> = {}'.format( open_deriv_data, (2-p0)/surv_time_mean ) )

		# plt.figure(1)
		# plt.loglog( params_model.loc['optimal', 'pnu'], params_model_mle.loc['optimal', 'pnu'], 'o' )
		# plt.axis([0, 1, 0, 1])
		# plt.figure(2)
		# plt.loglog( params_model.loc['optimal', 'ptau'], params_model_mle.loc['optimal', 'ptau'], 'o' )
		# plt.axis([0, 1, 0, 1])
		# plt.show()

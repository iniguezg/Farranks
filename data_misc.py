#! /usr/bin/env python

#Farranks - Exploring rank dynamics in complex systems
#Copyright (C) 2022 Gerardo Iñiguez

### MODULE FOR MISCELLANEOUS FUNCTIONS FOR DATA IN FARRANKS PROJECT ###

#import modules
import numpy as np
import pandas as pd
import pickle as pk
import random as rn
import itertools as it
from datetime import datetime
import sklearn.metrics as skm

import props_misc, model_misc


## FUNCTIONS ##

#function to get parameters for all datasets
def data_params( datasets, location, loadflag, saveloc ):
	"""Get parameters for all datasets"""

	#get dataframe of parameters for all datasets
	savename = saveloc+'params_data.pkl' #savename
	if loadflag == 'y': #load file
		params_data = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute parameters

		#initialise dataframe of parameters for datasets
		params_data = pd.DataFrame( np.zeros( ( len(datasets), 3 ), dtype=int ), index=pd.Series( datasets, name='dataset') , columns=pd.Series( [ 'N', 'N0', 'T' ], name='parameter' ) )

		for dataname in datasets: #loop through considered datasets

			print( 'dataset name: ' + dataname ) #print output

			loc_data = location + dataname + '/data/' #data location

			#load list of times
			timelist = pd.read_csv( location+dataname+'/time_list', header=None, names=['time'] )

			#look for all elements in the system across time

			all_elements = set() #initialise set of elements in system

			for post, t in timelist.itertuples(): #loop through (time position, time) tuples

				#get ranking at time t (elements and corresponding scores at given time)
				ranking = pd.read_csv( loc_data+str( t ), sep='\s+', header=None, names=[ 'element', 'score' ], converters={ 'element' : str } )
				#drop duplicate elements (disregarding score) (just in case)
				ranking.drop_duplicates( subset='element', inplace=True )

				all_elements.update( ranking.element ) #add elements to system set

			#get system size, ranking size, and number of time intervals
			params_data.at[ dataname, 'N'] = len( all_elements )
			params_data.at[ dataname, 'N0'] = len( ranking )
			params_data.at[ dataname, 'T'] = len( timelist )

		params_data.to_pickle( savename ) #save dataframe to file

	return params_data


#function to get average duration of time interval (in days) for all datasets
def data_intervals( datasets, formats, location, loadflag, saveloc ):
	"""Get average duration of time interval (in days) for all datasets"""

	#get dataframe of interval durations for all datasets
	savename = saveloc+'intervals_data.pkl' #savename
	if loadflag == 'y': #load file
		intervals_data = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute interval durations

		#initialise dataframe of parameters for datasets
		intervals_data = pd.DataFrame( np.zeros(( len(datasets), 1 )), index=pd.Series( datasets, name='dataset') , columns=pd.Series( [ 'tdays' ], name='parameter' ) )

		for dataname in datasets: #loop through considered datasets

			print( 'dataset name: ' + dataname ) #print output

			loc_data = location + dataname + '/data/' #data location

			#load list of times
			timelist = pd.read_csv( location+dataname+'/time_list', header=None, names=['time'] )

			#go through exceptions
			for t in range( len(timelist) ): #loop through time intervals [0, ..., T-1]
				if dataname == 'Virus': #add 1 and padded zeros
					timelist.iat[ t, 0 ] = timelist.iat[ t, 0 ][ :4 ] + str( int( timelist.iat[ t, 0 ][ 4: ] ) + 1 ).zfill( 3 )

			duration = 0.0 #initialise average duration of time interval (in days)

			for t in range( 1, len(timelist) ): #loop through time intervals [1, ..., T-1]

				#previous/current time strings (column 0 is time in timelist)
				prev_tstr = str( timelist.iat[ t - 1, 0 ] )
				tstr = str( timelist.iat[ t, 0 ] )

				#previous/current time objects
				prev_time = datetime.strptime( prev_tstr, formats[ dataname ] )
				time = datetime.strptime( tstr, formats[ dataname ] )

				delta = abs( time - prev_time ) #get time difference object

				duration += delta.days #accumulate difference (in days)
			duration /= len(timelist) - 1 #and average!

			intervals_data.at[ dataname, 'tdays' ] = duration #store duration of time interval

		intervals_data.to_pickle( savename ) #save dataframe to file

	return intervals_data


#function to get rank/element time series for given dataset
def data_series( dataname, params, location, loadflag, saveloc ):
	"""Get rank/element time series for all datasets"""

	N, N0, T = params['N'], params['N0'], params['T'] #get parameters

	#filenames for output files
	param_str = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )
	savenames = ( saveloc + 'rankseries_' + param_str, saveloc + 'elemseries_' + param_str )

	if loadflag == 'y': #load files
		rankseries = pd.read_pickle( savenames[0] )
		elemseries = pd.read_pickle( savenames[1] )

	elif loadflag == 'n': #or else, compute series

		loc_data = location + dataname + '/data/' #data location

		#load list of times
		timelist = pd.read_csv( location+dataname+'/time_list', header=None, names=['time'] )

		#first, we look for all elements in the system across time!

		all_elements = set() #initialise set of elements in system
		for t, tlabel in timelist.itertuples(): #loop through (time, time label) tuples

			#get ranking at time t (elements and corresponding scores at given time)
			ranking = pd.read_csv( loc_data+str( tlabel ), sep='\s+', header=None, names=[ 'element', 'score' ], converters={ 'element' : str } )
			ranking.drop_duplicates( subset='element', inplace=True ) #drop duplicate elements (disregarding score) (just in case)

			all_elements.update( ranking.element ) #add elements to system set

		#initialise rank time series with the full set of elements
		#time = 0, ..., T-1, elements = N (sorted) names in data
		rankseries = pd.DataFrame( np.zeros(( T, N ))*np.nan, index=pd.Series( range(T), name='time' ), columns=pd.Series( sorted(all_elements), name='element' ) )

		#initialise element time series with ranks (in ranking!)
		#time = 0, ..., T-1, ranks = 0, ..., N0 - 1
		elemseries = pd.DataFrame( np.empty( ( T, N0 ), dtype=str ), index=pd.Series( range(T), name='time' ), columns=pd.Series( range(N0), name='rank' ) )

		#then we fill the rank/element time series with ranks (in ranking!)

		for t, tlabel in timelist.itertuples(): #loop through (time, time label) tuples

			print( 't, t label = {}, {}'.format( t, tlabel ) ) #to know where we are

			#get ranking at time t (elements and corresponding scores at given time)
			ranking = pd.read_csv( loc_data+str( tlabel ), sep='\s+', header=None, names=[ 'element', 'score' ], converters={ 'element' : str } )
			ranking.drop_duplicates( subset='element', inplace=True ) #drop duplicate elements (disregarding score) (just in case)

			#sort ranking
			ranking.sort_values( 'score', axis='index', ascending=False, inplace=True )
			ranking.reset_index( drop=True, inplace=True ) #and reset index to new sorted ranking

			#loop through rank (i.e. index = 0, ..., N0 - 1), element and score
			for rank, elem, score in ranking.itertuples():
				rankseries.at[ t, elem ] = rank #store current rank of element
				elemseries.at[ t, rank ] = elem #store current element in rank

		rankseries.to_pickle( savenames[0] ) #save series to files
		elemseries.to_pickle( savenames[1] )

	return rankseries, elemseries


#function to (sub)sample rank/element time series / parameters for given dataset
def data_params_sample( dataname, params, location, loadflag, saveloc ):
	"""(Sub)sample rank/element time series / parameters for given dataset"""

	#get dataframe of parameters for all dataset samples
	savename = saveloc+'params_sample_{}.pkl'.format( dataname ) #savename
	if loadflag == 'y': #load file
		params_sample = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute series

		N, N0, T = params['N'], params['N0'], params['T'] #get parameters of original dataset

		#load original rank/element time series
		param_str = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )
		rankseries = pd.read_pickle( location + 'rankseries_' + param_str )
		elemseries = pd.read_pickle( location + 'elemseries_' + param_str )

		#initialise dataframe of parameters for dataset samples
		params_sample = pd.DataFrame( np.zeros( ( T-1, 3 ), dtype=int ), index=pd.Series( range(1, T), name='jump') , columns=pd.Series( [ 'N', 'N0', 'T' ], name='parameter' ) )
		params_sample.loc[1] = N, N0, T #fill up original dataset parameters (jump=1)

		for jump in range( 2, T ): #loop through (all possible) jump values

			#sample element/rank time series (drop elements disappearing from ranking!)
			elemseries_sample = elemseries[::jump]
			rankseries_sample = rankseries[::jump].dropna( axis=1, how='all' )

			#reset time index (to 0, ..., T_sample-1)
			elemseries_sample.reset_index( drop=True, inplace=True )
			rankseries_sample.reset_index( drop=True, inplace=True )

			#recalculate parameters for sampled dataset
			N_sample = rankseries_sample.columns.size #N minus elements without rank
			N0_sample = elemseries_sample.columns.size #same as N0
			T_sample = elemseries_sample.index.size #same as in sample rank time series

			#save sample rank/element time series
			param_str_sample = dataname+'_jump{}_N{}_N0{}_T{}.pkl'.format( jump, N_sample, N0_sample, T_sample )
			rankseries_sample.to_pickle( saveloc + 'rankseries_' + param_str_sample )
			elemseries_sample.to_pickle( saveloc + 'elemseries_' + param_str_sample )

			#update sample parameter dataframe
			params_sample.loc[jump] = N_sample, N0_sample, T_sample

		#and save it to file
		params_sample.to_pickle( savename )

	return params_sample


#function to get score time series for given dataset
def data_scores( dataname, params, location, loadflag, saveloc ):
	"""Get score time series for all datasets"""

	N, N0, T = params['N'], params['N0'], params['T'] #get parameters

	#filenames for output files
	param_str = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )
	savename = saveloc + 'scorseries_' + param_str

	if loadflag == 'y': #load files
		scorseries = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute series

		loc_data = location + dataname + '/data/' #data location

		#load list of times
		timelist = pd.read_csv( location+dataname+'/time_list', header=None, names=['time'] )

		#initialise score time series with (observable) set of scores
		#time = 0, ..., T-1, ranks = 0, ..., N0-1
		scorseries = pd.DataFrame( np.zeros(( T, N0 )), index=pd.Series( range(T), name='time' ), columns=pd.Series( range(N0), name='rank' ) )

		#then we fill the time series with all scores!

		for t, tlabel in timelist.itertuples(): #loop through (time, time label) tuples

			print( 't, t label = {}, {}'.format( t, tlabel ) ) #to know where we are

			#get ranking at time t (elements and corresponding scores at given time)
			ranking = pd.read_csv( loc_data+str( tlabel ), sep='\s+', header=None, names=[ 'element', 'score' ], converters={ 'element' : str } )
			ranking.drop_duplicates( subset='element', inplace=True ) #drop duplicate elements (disregarding score) (just in case)

			#sort ranking
			ranking.sort_values( 'score', axis='index', ascending=False, inplace=True )
			ranking.reset_index( drop=True, inplace=True ) #and reset index to new sorted ranking

			scorseries.loc[ t ] = ranking.score #store scores in current ranks

		scorseries.to_pickle( savename ) #save series to file

	return scorseries


#function to get flux/rank/open/succ/mle/tran properties for rank/element time series of given dataset
def data_props( prop_name, dataname, params, location, loadflag, saveloc ):
	"""Get flux/rank/open/succ/mle/tran properties for rank/element time series of given dataset"""

	#filename for output files
	param_str = dataname+'_N{}_N0{}_T{}.pkl'.format( params['N'], params['N0'], params['T'] )
	if prop_name == 'flux':
		savenames = ( saveloc + 'fluOprops_' + param_str, saveloc + 'fluIprops_' + param_str )
	if prop_name == 'rank':
		savenames = ( saveloc + 'rankprops_' + param_str, )
	if prop_name == 'open':
		savenames = ( saveloc + 'openprops_' + param_str, )
	if prop_name == 'disp':
		savenames = ( saveloc + 'dispprops_' + param_str, )
	if prop_name == 'succ':
		savenames = ( saveloc + 'success_' + param_str, saveloc + 'surprise_' + param_str )
	if prop_name == 'mle':
		savenames = ( saveloc + 'mleTprops_' + param_str, saveloc + 'mleDprops_' + param_str )
	if prop_name == 'tran':
		savenames = ( saveloc + 'tranprops_' + param_str, )

	#load/calculate properties
	if loadflag == 'y': #load files
		if prop_name == 'flux':
			fluOprops = pd.read_pickle( savenames[0] )
			fluIprops = pd.read_pickle( savenames[1] )
		if prop_name == 'rank':
			rankprops = pd.read_pickle( savenames[0] )
		if prop_name == 'open':
			openprops = pd.read_pickle( savenames[0] )
		if prop_name == 'disp':
			dispprops = pd.read_pickle( savenames[0] )
		if prop_name == 'succ':
			success = pd.read_pickle( savenames[0] )
			surprise = pd.read_pickle( savenames[1] )
		if prop_name == 'mle':
			mleTprops = pd.read_pickle( savenames[0] )
			mleDprops = pd.read_pickle( savenames[1] )
		if prop_name == 'tran':
			tranprops = pd.read_pickle( savenames[0] )

	elif loadflag == 'n': #or else, compute properties

		#get rank/element time series for dataset
		rankseries, elemseries = data_series( dataname, params, location, 'y', saveloc ) #(assume loadflag = y)

		if prop_name == 'flux':
			#get flux properties for rank/element time series of data
			fluOprops, fluIprops = props_misc.get_flux_props( rankseries, elemseries, params )
			fluOprops.to_pickle( savenames[0] ) #and save results
			fluIprops.to_pickle( savenames[1] )

		if prop_name == 'rank':
			#get rank properties for rank/element time series of data
			rankprops = props_misc.get_rank_props( rankseries, elemseries, params )
			rankprops.to_pickle( savenames[0] ) #and save results

		if prop_name == 'open':
			#get open properties for rank/element time series of data
			openprops = props_misc.get_open_props( rankseries, elemseries, params )
			openprops.to_pickle( savenames[0] ) #and save results

		if prop_name == 'disp':
			#get disp properties for rank/element time series of data
			dispprops = props_misc.get_disp_props( rankseries, elemseries, params )
			dispprops.to_pickle( savenames[0] ) #and save results

		if prop_name == 'succ':
			#get succ properties for rank/element time series of data
			success, surprise = props_misc.get_succ_props( rankseries, elemseries, params )
			success.to_pickle( savenames[0] ) #and save results
			surprise.to_pickle( savenames[1] )

		if prop_name == 'mle':
			#get MLE properties for rank/element time series of data
			mleTprops, mleDprops = props_misc.get_MLE_props( rankseries, elemseries, params )
			mleTprops.to_pickle( savenames[0] ) #and save results
			mleDprops.to_pickle( savenames[1] )

		if prop_name == 'tran':
			#get tran properties for rank/element time series of data
			tranprops = props_misc.get_tran_props( rankseries, elemseries, params )
			tranprops.to_pickle( savenames[0] ) #and save results

	if prop_name == 'flux':
		return fluOprops, fluIprops
	if prop_name == 'rank':
		return rankprops
	if prop_name == 'open':
		return openprops
	if prop_name == 'disp':
		return dispprops
	if prop_name == 'succ':
		return success, surprise
	if prop_name == 'mle':
		return mleTprops, mleDprops
	if prop_name == 'tran':
		return tranprops


#function to get model phase diagram for given dataset (theo)
def data_estimate_PD_theo( dataname, params, loadflag, saveloc_data, par_step=0.01, gof_str='MSE', datatype='open' ):
	"""Get model phase diagram for given dataset (theo)"""

	N, N0, T = params['N'], params['N0'], params['T'] #parameters from data

	#explored parameters
	ptau_vals = np.arange( 0, 1 + par_step/2, par_step ) #diffusion probability
	pnu_vals = np.arange( 0, 1 + par_step/2, par_step ) #replacement probability

	#properties to fit
	if datatype == 'open':
		prop_names = { 'flux_time' : 'flux',
				   'openness' : 'open',
				   'flux_rank' : 'flux',
				   'change' : 'rank',
				   'success' : 'succ' }
	elif datatype == 'closed':
		prop_names = { 'change' : 'rank',
				   'success' : 'succ' }

	thres = 0.5 #threshold to calculate success/surprise measure

	#filename for model output file
	#default Goodness-of-fit measure: Mean Squared Error (string: MSE)
	param_str_model = dataname+'_{}.pkl'.format( gof_str )

	if loadflag == 'y': #load files
		model_PD = pk.load( open( saveloc_data + 'modelPD_' + param_str_model, 'rb' ) )

	elif loadflag == 'n': #or else, compute phase diagrams
		model_PD = {} #initialise dict of phase diagrams by rank property

		param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T ) #filename for data

		for prop_name in prop_names: #loop through rank measures to fit
			print( '\trank property: '+prop_name )

			#prepare data

			if prop_name == 'flux_time':
				fluOprops_data = pd.read_pickle( saveloc_data + 'fluOprops_' + param_str_data )
				prop_data = fluOprops_data.mean( axis=1 ) #OUT-flow (=IN-flow when averaged over Nt)

			if prop_name == 'openness':
				openprops_data = pd.read_pickle( saveloc_data + 'openprops_' + param_str_data )
				prop_data = openprops_data.loc[ 'openness' ]

			if prop_name == 'flux_rank':
				fluOprops_data = pd.read_pickle( saveloc_data + 'fluOprops_' + param_str_data )
				prop_data = fluOprops_data.mean( axis=0 ) #OUT-flow (averaged over time t)

			if prop_name == 'change':
				rankprops_data = pd.read_pickle( saveloc_data + 'rankprops_' + param_str_data )
				prop_data = rankprops_data.loc[ 'rankchange' ]

			if prop_name == 'success':
				success_data = pd.read_pickle( saveloc_data + 'success_' + param_str_data )
				prop_data = success_data.loc[ thres ]

			#prepare model

			model_PD[ prop_name ] = pd.DataFrame( np.ones(( len(pnu_vals), len(ptau_vals) ))*np.nan, index=pd.Series( np.flip(pnu_vals), name=r'$p_{\nu}$' ), columns=pd.Series( ptau_vals, name=r'$p_{\tau}$' ) ) #initialise dataframe of goodness-of-fit values (w/ flipped index!)

			for pnu, ptau in it.product( pnu_vals, ptau_vals ): #loop through parameteer values
				params[ 'pnu' ], params[ 'ptau' ] = pnu, ptau #set parameters for model (NOTE: params must be dict!)

				if prop_name == 'flux_time':
					prop_model = model_misc.flux_theo( params ) * np.ones(( T - 1 )) #constant flow over time

				if prop_name == 'openness':
					prop_model = model_misc.openness_theo( params )

				if prop_name == 'flux_rank':
					prop_model = model_misc.flux_out_theo( params )

				if prop_name == 'change':
					prop_model = model_misc.change_theo( params )

				if prop_name == 'success':
					prop_model = model_misc.success_theo( thres, params )

				#compute goodness-of-fit metric
				if gof_str == 'MSE':
					model_PD[ prop_name ].at[ pnu, ptau ] = skm.mean_squared_error( prop_data, prop_model )
				else:
					'goodness-of-fit metric not recognised!'

		#save to file!
		pk.dump( model_PD, open( saveloc_data + 'modelPD_' + param_str_model, 'wb' ) ) #save to file

	return model_PD


#function to get optimal model parameters for given dataset (theo)
def data_estimate_params_theo( dataname, params, loadflag, saveloc_data, gof_str='MSE', datatype='open' ):
	"""Get model optimal parameters for given dataset (theo)"""

	#properties to fit
	if datatype == 'open':
		prop_names = [ 'flux_time', 'openness', 'flux_rank', 'change', 'success' ]
	elif datatype == 'closed':
		prop_names = [ 'change', 'success' ]

	#filename for model output file
	#default Goodness-of-fit measure: Mean Squared Error (string: MSE)
	param_str_model = dataname+'_{}.pkl'.format( gof_str )

	if loadflag == 'y': #load files
		params_model = pd.read_pickle( saveloc_data + 'params_model_' + param_str_model )

	elif loadflag == 'n': #or else, compute phase diagrams
		params_model = pd.DataFrame( np.zeros( ( len(prop_names), 2 ) ), index=pd.Series( prop_names, name='property'), columns=pd.Series( [ 'pnu', 'ptau' ], name='parameter' ) )

		#load already-calculated model phase diagram!
		model_PD = pk.load( open( saveloc_data + 'modelPD_' + param_str_model, 'rb' ) )

		for prop_name in prop_names: #loop through rank measures to fit
			print( '\trank property: '+prop_name )

			#find (global) minimum in metric
			params_model.loc[ prop_name, : ] = model_PD[ prop_name ].stack().idxmin()

		#save to file!
		params_model.to_pickle( saveloc_data + 'params_model_' + param_str_model )

	return params_model

#function to get optimal model parameters for given dataset (all parameters at same time)
def data_estimate_params_all( dataname, params, loadflag, saveloc, datatype='open' ):
	"""Get optimal model parameters for given dataset (all parameters at same time)"""

	#filename for model output file
	param_str_model = dataname+'.pkl'

	if loadflag == 'y': #load files
		params_model = pd.read_pickle( saveloc + 'params_model_' + param_str_model )

	elif loadflag == 'n': #or else, solve system of equations for parameters
		params_model = pd.DataFrame( np.zeros( ( 1, 6 ) ), index=pd.Series( 'optimal', name='value'), columns=pd.Series( [ 'flux', 'open_deriv', 'success', 'p0', 'ptau', 'pnu' ], name='parameter' ) )

		#get optimal model parameters from flux/open/succ properties (all parameters at same time)
		params_model.loc[ 'optimal' ] = model_misc.estimate_params_all( dataname, params, saveloc, datatype=datatype )

		#save to file!
		params_model.to_pickle( saveloc + 'params_model_' + param_str_model )

	return params_model


#function to estimate parameter deviations for given dataset (all parameters at same time)
def data_estimate_params_devs( dataname, params, loadflag, saveloc, datatype='open' ):
	"""Estimate parameter deviations for given dataset (all parameters at same time)"""

	param_str = dataname+'.pkl' #filename for output files

	if loadflag == 'y': #load files
		params_devs = pd.read_pickle( saveloc + 'params_devs_' + param_str )

	elif loadflag == 'n': #or else, compute deviations

		ntimes = params['ntimes'] #number of realisations (for bootstrapping)
		params_model = pd.read_pickle( saveloc + 'params_model_' + param_str ) #fitted parameters of dataset

		#set props and params for each bootstrap realisation
		prop_names = ( 'flux', 'open', 'succ' ) #properties to compute
		params_rel = { 'N':params['N'], 'N0':params['N0'], 'T':params['T'], 'ptau':params_model.loc['optimal', 'ptau'], 'pnu':params_model.loc['optimal', 'pnu'], 'ntimes':1 } #parameters to run model
		print( '\tN = {}'.format( params_rel['N'] ), flush=True )

		#estimate system size from bootstrapped realisations of model and use as guess for N
		N_est_theo = model_misc.estimate_params_size( dataname, params, 'y', saveloc )
		print( '\tN_est_theo = {}'.format( N_est_theo ), flush=True )

		#approx guess from theory: estimate system size from rank openness in model
		print( '\t(approx guess from theory = {})'.format( model_misc.N_est_theo( params_rel ) ), flush=True )

		#initialise dataframe of parameters in bootstrapped model (to compute deviations)
		params_devs = pd.DataFrame( np.zeros( ( ntimes, 6 ) ), index=pd.Series( range(ntimes), name='realisation'), columns=pd.Series( [ 'flux', 'open_deriv', 'success', 'p0', 'ptau', 'pnu' ], name='parameter' ) )

		for nt in range(ntimes): #loop through realisations
			print( 'nt = {}'.format(nt), flush=True ) #to know where we stand

			#estimate system size from rank openness in model (theo) and use as guess for N
			params_rel['N'] = N_est_theo

			prop_dict = model_misc.model_props( prop_names, params_rel ) #run model
			params_rel['N'] = prop_dict['params']['N'] #update varying parameters
			print( '\t\tN = {}'.format( params_rel['N'] ) )

			params_devs.loc[ nt ] = model_misc.estimate_params_all( dataname, params_rel, saveloc, prop_dict=prop_dict, datatype=datatype )

			#save every realisation (for large datasets in cluster)
			params_devs.to_pickle( saveloc + 'params_devs_' + param_str )

	return params_devs


#function to get optimal model parameters for samples of given dataset (all parameters at same time)
def data_estimate_params_sample( dataname, params, location, loadflag, saveloc, datatype='open' ):
	"""Get optimal model parameters for samples of given dataset (all parameters at same time)"""

	#filename for model output file
	param_str_model = dataname+'.pkl'

	if loadflag == 'y': #load files
		params_model = pd.read_pickle( saveloc + 'params_model_' + param_str_model )

	elif loadflag == 'n': #or else, solve system of equations for parameters

		T = params['T'] #get parameters

		#get parameters for all datasets, given dataset, and sample dataset
		params_sample = pd.read_pickle( saveloc+'params_sample_{}.pkl'.format( dataname ) )

		#initialise dataframe of optimal model parameters
		params_model = pd.DataFrame( np.zeros(( T-1, 6 )), index=pd.Series( range(1, T), name='jump'), columns=pd.Series( [ 'flux', 'open_deriv', 'success', 'p0', 'ptau', 'pnu' ], name='parameter' ) )

		#get optimal model parameters for original dataset
		params_model.loc[ 1 ] = model_misc.estimate_params_all( dataname, params, location, datatype=datatype )

		for jump in range( 2, T ): #loop through (all possible) jump values
			# print( '\tjump: {}'.format(jump) ) #print jump
			dataname_sample = dataname + '_jump{}'.format( jump ) #pick sample
			params_sample_jump = params_sample.loc[ jump ] #and get its parameters

			#get optimal model parameters for sample dataset
			params_model.loc[ jump ] = model_misc.estimate_params_all( dataname_sample, params_sample_jump, saveloc, datatype=datatype )

		#save to file!
		params_model.to_pickle( saveloc + 'params_model_' + param_str_model )

	return params_model


#function to get optimal model parameters for given dataset (maximum likelihood estimation)
def data_estimate_params_MLE( dataname, params, loadflag, saveloc_data, datatype='open', sample_frac=False, saveflag='n' ):
	"""Get optimal model parameters for given dataset (maximum likelihood estimation)"""

	#filename for model output file
	param_str_model = dataname+'.pkl'

	if loadflag == 'y': #load files
		params_model = pd.read_pickle( saveloc_data + 'params_model_mle_' + param_str_model )

	elif loadflag == 'n': #or else, solve system of equations for parameters
		params_model = pd.DataFrame( np.zeros( ( 1, 2 ) ), index=pd.Series( 'optimal', name='value'), columns=pd.Series( [ 'pnu', 'ptau' ], name='parameter' ) )

		#get optimal model parameters from mle properties (maximum likelihood estimation)
		ptau_star, pnu_star = model_misc.estimate_params_MLE( dataname, params, saveloc_data, datatype=datatype, sample_frac=sample_frac )

		#save to file!
		params_model.loc[ 'optimal', [ 'pnu', 'ptau' ] ] = pnu_star, ptau_star
		if saveflag == 'y':
			params_model.to_pickle( saveloc_data + 'params_model_mle_' + param_str_model )

	return params_model


#function to get model phase diagram for given dataset
def data_estimate_PD( dataname, params, loadflag, saveloc_data, saveloc_model ):
	"""Get model phase diagram for given dataset"""

	N, N0, T = params['N'], params['N0'], params['T'] #parameters from data
	ntimes = params['ntimes'] #number of realisations to average

	ptau_vals = np.arange( 0, 1.01, 0.02 ) #explored parameters
	pnu_vals = np.arange( 0, 1.01, 0.02 )
#	pnu_vals = np.array([ 0. ])
	prop_names = { 'flux_time' : 'flux',
				   'openness' : 'open',
				   'flux_rank' : 'flux',
				   'change' : 'rank',
				   'diversity' : 'rank',
				   'success' : 'succ' } #properties to fit

	thres = 0.5 #threshold to calculate success/surprise measure

	#filename for model output file. Goodness-of-fit measure: Mean Squared Error
	param_str_model = dataname+'_MSE_ntimes{}.pkl'.format( ntimes )

	if loadflag == 'y': #load files
		model_PD = pk.load( open( saveloc_data + 'modelPD_' + param_str_model, 'rb' ) )

	elif loadflag == 'n': #or else, compute phase diagrams
		model_PD = {} #initialise dict of phase diagrams by rank property

		param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T ) #filename for data

		for prop_name in prop_names: #loop through rank measures to fit
			print( '\trank property: '+prop_name )

			#prepare data

			if prop_name == 'flux_time':
				fluIprops_data = pd.read_pickle( saveloc_data + 'fluIprops_' + param_str_data )
				prop_data = fluIprops_data.mean( axis=1 ) #IN-flow (=OUT-flow when averaged over Nt)

			if prop_name == 'openness':
				openprops_data = pd.read_pickle( saveloc_data + 'openprops_' + param_str_data )
				prop_data = openprops_data.loc[ 'openness' ]

			if prop_name == 'flux_rank':
				fluOprops_data = pd.read_pickle( saveloc_data + 'fluOprops_' + param_str_data )
				prop_data = fluOprops_data.mean( axis=0 ) #OUT-flow (averaged over time t)

			if prop_name == 'change':
				rankprops_data = pd.read_pickle( saveloc_data + 'rankprops_' + param_str_data )
				prop_data = rankprops_data.loc[ 'rankchange' ]

			if prop_name == 'diversity':
				rankprops_data = pd.read_pickle( saveloc_data + 'rankprops_' + param_str_data )
				diversity_data = rankprops_data.loc[ 'diversity' ]
				prop_data = diversity_data / diversity_data.max() #normalise by max diversity!

			if prop_name == 'success':
				success_data = pd.read_pickle( saveloc_data + 'success_' + param_str_data )
				prop_data = success_data.loc[ thres ]

			#prepare model

			model_PD[ prop_name ] = pd.DataFrame( np.empty(( len(pnu_vals), len(ptau_vals) ))*np.nan, index=pd.Series( np.flip(pnu_vals), name=r'$p_{\nu}$' ), columns=pd.Series( ptau_vals, name=r'$p_{\tau}$' ) ) #initialise dataframe of goodness-of-fit values (w/ flipped index!)

			for pnu, ptau in it.product( pnu_vals, ptau_vals ): #loop through parameteer values
				params[ 'pnu' ], params[ 'ptau' ] = pnu, ptau #set parameters for model (NOTE: params must be dict!)

				try: #skip if model files are still missing
					prop_dict = model_misc.model_props( [ prop_names[prop_name] ], params, 'y', saveloc_model ) #load model (assume loadflag = y)

					if prop_name == 'flux_time':
						fluOprops_model, fluIprops_model = prop_dict['flux']
						prop_model = fluIprops_model.mean( axis=1 ) #IN-flow (=OUT-flow when averaged over Nt)

					if prop_name == 'openness':
						openprops_model, = prop_dict['open']
						prop_model = openprops_model.loc[ 'openness' ]

					if prop_name == 'flux_rank':
						fluOprops_model, fluIprops_model = prop_dict['flux']
						prop_model = fluOprops_model.mean( axis=0 ) #OUT-flow (averaged over time t)

					if prop_name == 'change':
						rankprops_model, = prop_dict['rank']
						prop_model = rankprops_model.loc[ 'rankchange' ]

					if prop_name == 'diversity':
						rankprops_model, = prop_dict['rank']
						diversity_model = rankprops_model.loc[ 'diversity' ]
						prop_model = diversity_model / diversity_model.max() #normalise by max diversity!

					if prop_name == 'success':
						success_model, surprise_model = prop_dict['succ']
						prop_model = success_model.loc[ thres ]

					#compute goodness-of-fit metric
					model_PD[ prop_name ].at[ pnu, ptau ] = skm.mean_squared_error( prop_data, prop_model )

				except FileNotFoundError:
					print( 'missing model files at pnu, ptau = {:.2f}, {:.2f}'.format( pnu, ptau ) )

		#save to file!
		pk.dump( model_PD, open( saveloc_data + 'modelPD_' + param_str_model, 'wb' ) ) #save to file

	return model_PD


#function to get model optimal parameters for given dataset
def data_estimate_params( dataname, params, loadflag, saveloc_data, saveloc_model ):
	"""Get model optimal parameters for given dataset"""

	ntimes = params['ntimes'] #number of realisations to average
	prop_names = [ 'flux_time', 'openness', 'flux_rank', 'change', 'diversity', 'success' ] #properties to fit

	#filename for model output file. Goodness-of-fit measure: Mean Squared Error
	param_str_model = dataname+'_MSE_ntimes{}.pkl'.format( ntimes )

	if loadflag == 'y': #load files
		params_model = pd.read_pickle( saveloc_data + 'params_model_' + param_str_model )

	elif loadflag == 'n': #or else, compute phase diagrams
		params_model = pd.DataFrame( np.zeros( ( len(prop_names), 2 ) ), index=pd.Series( prop_names, name='property'), columns=pd.Series( [ 'pnu', 'ptau' ], name='parameter' ) )

		#load already-calculated model phase diagram!
		model_PD = pk.load( open( saveloc_data + 'modelPD_' + param_str_model, 'rb' ) )

		for prop_name in prop_names: #loop through rank measures to fit
			print( '\trank property: '+prop_name )

#			#find (global) minimum in metric
#			params_model.loc[ prop_name, : ] = model_PD[ prop_name ].stack().idxmin()

			#find (local) optimal parameters ONLY along ptau = pnu diagonal!
			ptau_vals = model_PD[ prop_name ].columns #ptau values
			model_PD_diag = [ model_PD[ prop_name ].loc[ ptau, ptau ] for ptau in ptau_vals ]
			params_model.loc[ prop_name, : ] = ptau_vals[ np.nanargmin( model_PD_diag ) ]

		#save to file!
		params_model.to_pickle( saveloc_data + 'params_model_' + param_str_model )

	return params_model


## RUNNING DATA MISC MODULE ##

if __name__ == "__main__":

	print( 'MODULE FOR MISCELLANEOUS FUNCTIONS FOR DATA IN FARRANKS PROJECT' )

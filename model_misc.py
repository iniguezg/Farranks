#! /usr/bin/env python

### MODULE FOR MISCELLANEOUS FUNCTIONS FOR NULL MODEL IN FARRANKS PROJECT ###

#import modules
import random
import numpy as np
import pandas as pd
import mpmath as mpm
import itertools as it
import scipy.stats as st
import scipy.special as ss
import scipy.optimize as spo

import props_misc


## FUNCTIONS ##

#function to run null model of rank dynamics, according to parameters
def null_model( params ):
	"""Run null model of rank dynamics, according to parameters"""

	N, N0, T = params['N'], params['N0'], params['T'] #parameters from data
	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit

	#initialise element time series with ranks (in ranking!)
	#time = 0, ..., T-1, ranks = 0, ..., N0 - 1
	elemseries = pd.DataFrame( np.empty( ( T, N0 ), dtype=int ), index=pd.Series( range(T), name='time' ), columns=pd.Series( range(N0), name='rank' ) )
	elemseries.loc[ 0 ] = elemseries.columns #set initial condition at t=0 (using arbitrary element names)

	#initialise system at time t (same as elemseries for t=0)
	#ranks = 0, ..., N - 1 (0 = highest rank, N - 1 = lowest rank)
	system = list( range( N ) ) #position = rank, value = element
	Nnew = N #initialise counter of newly introduced elements

	# print( 'initial state:' )
	# print( 't 0, minit -, N0 {}, N {}, rank -, elem -, new_rank - : {}'.format( N0, N, [ R for R in system ] ) ) #to know where we are

	#rank dynamics
	for t in range( 1, T ): #loop through time intervals [1, ..., T-1]
#		print( 't = {}'.format( t ) ) #to know where we stand
		for minit in range( N ): #and N updates inside time intervals

			#dynamics of rank change
			if random.random() < ptau:

				rank = random.randint( 0, N - 1 ) #pick random rank R = 0, ..., N-1 (of whole system!)
				elem = system.pop( rank ) #take out element at selected rank
				new_rank = random.randint( 0, N - 1 ) #pick new rank R = 0, ..., N-1 at random (also N options!)
				system.insert( new_rank, elem ) #insert element at new rank

				# print( 'ptau dynamics:' )
				# print( 't {}, minit {}, N0 {}, N {}, rank {}, elem {}, new_rank {} : {}'.format( t, minit, N0, N, rank, elem, new_rank, [ R for R in system ] ) ) #to know where we are

			#dynamics of new elements
			if random.random() < pnu:

				rank = random.randint( 0, N - 1 ) #pick random rank R = 0, ..., N-1 (of whole system!)
				system[ rank ] = Nnew #substitute element at rank by new one
				Nnew += 1 #increase element counter

				# print( 'pnu dynamics:' )
				# print( 't {}, minit {}, N0 {}, N {}, rank {}, elem {} : {}'.format( t, minit, N0, N, rank, Nnew-1, [ R for R in system ] ) ) #to know where we are

		#save (current) elements in ranking (R = 0, ..., N0 - 1)
		elemseries.loc[ t ] = system[ :N0 ]

	#initialise rank time series with the full set of elements
	#time = 0, ..., T-1, elements = 0, ..., Nnew - 1
	rankseries = pd.DataFrame( np.zeros(( T, Nnew ))*np.nan, index=pd.Series( range(T), name='time' ), columns=pd.Series( range( Nnew ), name='element' ) )
	rankseries.loc[ 0, :N0-1 ] = rankseries.columns[ :N0 ] #set initial condition at t=0 (only ranking!

	for t in range( 1, T ): #loop through time intervals [1, ..., T-1]
		for rank, elem in enumerate( elemseries.loc[ t ] ): #and elements in each ranking
			rankseries.at[ t, elem ] = rank #store current rank of element

	#drop elements that have never been in ranking!
	rankseries = rankseries.drop( rankseries.columns[ rankseries.apply( lambda col: col.isnull().sum() == T ) ], axis=1 )

	#update parameters that have changed
	params_new = { 'N':rankseries.columns.size } #number of elements ever in ranking!

	return rankseries, elemseries, params_new


#function to get average flux/rank/open/succ/samp/mle properties for rank/element time series of null model
def model_props( prop_names, params, loadflag='n', saveloc='', saveflag='n' ):
	"""Get average flux/rank/open/succ/samp/mle properties for rank/element time series of null model"""

	ntimes = params['ntimes'] #number of realisations to average

	#filename for output files
	param_str = 'model_N{}_N0{}_T{}_ptau{:.2f}_pnu{:.2f}_ntimes{}.pkl'.format( params['N'], params['N0'], params['T'], params['ptau'], params['pnu'], ntimes )
	savenames = {} #initialise dict of savenames
	if 'flux' in prop_names:
		savenames['flux'] = ( saveloc + 'fluOprops_' + param_str, saveloc + 'fluIprops_' + param_str )
	if 'rank' in prop_names:
		savenames['rank'] = ( saveloc + 'rankprops_' + param_str, )
	if 'open' in prop_names:
		savenames['open'] = ( saveloc + 'openprops_' + param_str, )
	if 'disp' in prop_names:
		savenames['disp'] = ( saveloc + 'dispprops_' + param_str, )
	if 'succ' in prop_names:
		savenames['succ'] = ( saveloc + 'success_' + param_str, saveloc + 'surprise_' + param_str )
	if 'samp' in prop_names:
		savenames['samp'] = ( saveloc + 'sampprops_' + param_str, )
	if 'mle' in prop_names:
		savenames['mle'] = ( saveloc + 'mleTprops_' + param_str, saveloc + 'mleDprops_' + param_str )

	#load/calculate properties
	if loadflag == 'y': #load files
		if 'flux' in prop_names:
			fluOprops = pd.read_pickle( savenames['flux'][0] )
			fluIprops = pd.read_pickle( savenames['flux'][1] )
		if 'rank' in prop_names:
			rankprops = pd.read_pickle( savenames['rank'][0] )
		if 'open' in prop_names:
			openprops = pd.read_pickle( savenames['open'][0] )
		if 'disp' in prop_names:
			dispprops = pd.read_pickle( savenames['disp'][0] )
		if 'succ' in prop_names:
			success = pd.read_pickle( savenames['succ'][0] )
			surprise = pd.read_pickle( savenames['succ'][1] )
		if 'samp' in prop_names:
			sampprops = pd.read_pickle( savenames['samp'][0] )
		if 'mle' in prop_names:
			mleTprops = pd.read_pickle( savenames[0] )
			mleDprops = pd.read_pickle( savenames[1] )

	elif loadflag == 'n': #or else, compute properties

		#run model for the first time
		rankseries, elemseries, params_new = null_model( params )
		if 'flux' in prop_names:
			fluOprops, fluIprops = props_misc.get_flux_props( rankseries, elemseries, params )
		if 'rank' in prop_names:
			rankprops = props_misc.get_rank_props( rankseries, elemseries, params )
		if 'open' in prop_names:
			openprops = props_misc.get_open_props( rankseries, elemseries, params )
		if 'disp' in prop_names:
#			dispprops = props_misc.get_disp_props( rankseries, elemseries, params )
			dispprops = props_misc.get_disp_time_props( rankseries, elemseries, params )
		if 'succ' in prop_names:
			success, surprise = props_misc.get_succ_props( rankseries, elemseries, params )
		if 'samp' in prop_names:
			sampprops = props_misc.get_samp_props( rankseries, elemseries, params )
		if 'mle' in prop_names:
			mleTprops, mleDprops = props_misc.get_MLE_props( rankseries, elemseries, params )

		#and average over remaining realisations
		for nt in range( ntimes - 1 ):
#			if nt % 100 == 0: #to know where we are
			print( 'nt = {}'.format( nt ) )

			#run model again and again
			rankseries, elemseries, params_one = null_model( params )
			params_new['N'] += params_one['N'] #acumulate varying parameters

			if 'flux' in prop_names:
				fluOprops_one, fluIprops_one = props_misc.get_flux_props( rankseries, elemseries, params )
				fluOprops += fluOprops_one
				fluIprops += fluIprops_one
			if 'rank' in prop_names:
				rankprops_one = props_misc.get_rank_props( rankseries, elemseries, params )
				rankprops += rankprops_one
			if 'open' in prop_names:
				openprops_one = props_misc.get_open_props( rankseries, elemseries, params )
				openprops += openprops_one
			if 'disp' in prop_names:
#				dispprops_one = props_misc.get_disp_props( rankseries, elemseries, params )
				dispprops_one = props_misc.get_disp_time_props( rankseries, elemseries, params )
				dispprops += dispprops_one
			if 'succ' in prop_names:
				success_one, surprise_one = props_misc.get_succ_props( rankseries, elemseries, params )
				success += success_one
				surprise += surprise_one
			if 'samp' in prop_names:
				sampprops_one = props_misc.get_samp_props( rankseries, elemseries, params )
				sampprops += sampprops_one
			if 'mle' in prop_names:
				mleTprops_one, mleDprops_one = props_misc.get_MLE_props( rankseries, elemseries, params )
				mleTprops += mleTprops_one
				mleDprops += mleDprops_one

		#get averages!
		params_new['N'] = int( params_new['N'] / float(ntimes) ) #first varying parameters
		if 'flux' in prop_names:
			fluOprops /= ntimes
			fluIprops /= ntimes
			if saveflag == 'y':
				fluOprops.to_pickle( savenames['flux'][0] ) #and save results
				fluIprops.to_pickle( savenames['flux'][1] )
		if 'rank' in prop_names:
			rankprops /= ntimes
			if saveflag == 'y':
				rankprops.to_pickle( savenames['rank'][0] ) #and save result
		if 'open' in prop_names:
			openprops /= ntimes
			if saveflag == 'y':
				openprops.to_pickle( savenames['open'][0] ) #and save result
		if 'disp' in prop_names:
			dispprops /= ntimes
			if saveflag == 'y':
				dispprops.to_pickle( savenames['disp'][0] ) #and save result
		if 'succ' in prop_names:
			success /= ntimes
			surprise /= ntimes
			if saveflag == 'y':
				success.to_pickle( savenames['succ'][0] ) #and save results
				surprise.to_pickle( savenames['succ'][1] )
		if 'samp' in prop_names:
			sampprops /= ntimes
			if saveflag == 'y':
				sampprops.to_pickle( savenames['samp'][0] ) #and save result
		if 'mle' in prop_names:
			mleTprops /= ntimes
			mleDprops /= ntimes
			if saveflag == 'y':
				mleTprops.to_pickle( savenames['mle'][0] ) #and save result
				mleDprops.to_pickle( savenames['mle'][1] )

	prop_dict = {} #initialise dict of properties to return
	if loadflag == 'n':
		prop_dict['params'] = params_new #save only when not loading
	if 'flux' in prop_names:
		prop_dict['flux'] = ( fluOprops, fluIprops )
	if 'rank' in prop_names:
		prop_dict['rank'] = ( rankprops, )
	if 'open' in prop_names:
		prop_dict['open'] = ( openprops, )
	if 'disp' in prop_names:
		prop_dict['disp'] = ( dispprops, )
	if 'succ' in prop_names:
		prop_dict['succ'] = ( success, surprise )
	if 'samp' in prop_names:
		prop_dict['samp'] = ( sampprops, )
	if 'mle' in prop_names:
		prop_dict['mle'] = ( mleTprops, mleDprops )

	return prop_dict


#function to compute rank flux in model (theo)
def flux_theo( params ):
	"""Compute rank flux in model (theo)"""

	N, N0 = params['N'], params['N0'] #parameters from data/model
	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit

	p0 = N0 / float( N ) #ranking fraction

	flux_theo = 1 - np.exp( -pnu ) * ( p0 + (1 - p0) * np.exp( -ptau ) )

	return flux_theo


#function to compute rank success in model (theo)
def success_theo( thres, params ):
	"""Compute rank success in model (theo)"""

	N, N0, T = params['N'], params['N0'], params['T'] #parameters from data/model
	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit

	p0 = N0 / float( N ) #ranking fraction

	lags = np.arange( T ) #possible time lags [0, ..., T-1]

	success_theo = np.exp( -pnu * lags ) * ( thres * p0 + (1 - thres * p0) * np.exp( -ptau * lags ) )

	return success_theo


#function to compute rank success through time in model (theo)
def success_time_theo( thres, t, params ):
	"""Compute rank success through time in model (theo)"""

	N, N0 = params['N'], params['N0'] #parameters from data/model
	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit

	p0 = N0 / float( N ) #ranking fraction

	success_theo = np.exp( -pnu * t ) * ( thres * p0 + (1 - thres * p0) * np.exp( -ptau * t ) )

	return success_theo


#function to compute rank openness in model (theo)
def openness_theo( params ):
	"""Compute rank openness in model (theo)"""

	N, N0, T = params['N'], params['N0'], params['T'] #parameters from data/model
	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit

	p0 = N0 / float( N ) #ranking fraction

	t_vals = np.linspace( 0, T-1, num=T ) #values of displaced rank

	exp_decay = np.exp( -( pnu + p0 * ptau ) * t_vals )

	if ptau * pnu > 0: #if anything happens:
		inf_value = ( pnu + ptau ) / ( pnu + p0 * ptau )
		openness_theo = ( 1 + pnu * t_vals ) * ( exp_decay + inf_value * ( 1 - exp_decay ) )
	else:
		openness_theo = np.ones( T ) #null openness

	return openness_theo


#function to compute rank openness through time in model (theo)
def openness_time_theo( params ):
	"""Compute rank openness through time in model (theo)"""

	N, N0, T = params['N'], params['N0'], params['T'] #parameters from data/model
	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit

	p0 = N0 / float( N ) #ranking fraction

	exp_decay = np.exp( -( pnu + p0 * ptau ) * (T-1) )
	inf_value = ( pnu + ptau ) / ( pnu + p0 * ptau )
	openness_theo = ( 1 + pnu * (T-1) ) * ( exp_decay + inf_value * ( 1 - exp_decay ) )

	return openness_theo


#function to compute rank openness derivative in model (theo)
def open_deriv_theo( params ):
	"""Compute rank openness derivative in model (theo)"""

	N, N0, T = params['N'], params['N0'], params['T'] #parameters from data/model
	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit

	p0 = N0 / float( N ) #ranking fraction

	exp_decay = np.exp( -( pnu + p0 * ptau ) * (T-1) )
	inf_value = ( pnu + ptau ) / ( pnu + p0 * ptau )

	open_deriv_theo = ( 1/(T-1) ) * ( ( 1 + pnu * (T-1) ) * ( exp_decay + inf_value * ( 1 - exp_decay ) ) - 1 )

	return open_deriv_theo


#function to estimate system size from rank openness in model (theo)
def N_est_theo( params ):
	"""Estimate system size from rank openness in model (theo)"""

	N, N0, T = params['N'], params['N0'], params['T'] #parameters from data/model
	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit

	p0 = N0 / float( N ) #ranking fraction

	exp_decay = np.exp( -( pnu + p0 * ptau ) * (T-1) )
	inf_value = ( pnu + ptau ) / ( pnu + p0 * ptau )

	N_est_theo = int( N / ( p0 * ( 1 + pnu*(T-1) ) * ( exp_decay + inf_value*(1 - exp_decay) ) ) )

	return N_est_theo


#function to compute rank displacement in model (theo)
def displacement_theo( r, params ):
	"""Compute rank displacement in model (theo)"""

	N, N0 = params['N'], params['N0'] #parameters from data/model
	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit

	dr = 1 / float( N ) #rank increment
	p0 = N0 / float( N ) #ranking fraction

	x_vals = np.linspace( dr, p0, num=N0 ) #values of displaced rank

	levi_sea = dr * ( 1 - np.exp( -ptau ) )
	peak_sdev = np.sqrt( 2 * ptau * dr * r * (1 - r) )
	disp_theo = np.exp( -pnu ) * ( levi_sea + np.exp( -ptau ) * st.norm.pdf( x_vals, loc=r, scale=peak_sdev ) * dr )

	return disp_theo


#function to compute rank displacement through time in model (theo)
def disp_time_theo( r, t, params ):
	"""Compute rank displacement through time in model (theo)"""

	N, N0 = params['N'], params['N0'] #parameters from data/model
	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit

	dr = 1 / float( N ) #rank increment
	p0 = N0 / float( N ) #ranking fraction

	x_vals = np.linspace( dr, p0, num=N0 ) #values of displaced rank

	levi_sea = dr * ( 1 - np.exp( -ptau * t ) )
	peak_sdev = np.sqrt( 2 * ptau * dr * r * (1 - r) * t )
	disp_theo = np.exp( -pnu * t ) * ( levi_sea + np.exp( -ptau * t ) * st.norm.pdf( x_vals, loc=r, scale=peak_sdev ) * dr )

	return disp_theo


#function to compute rank change in model (theo)
def change_theo( params ):
	"""Compute rank change in model (theo)"""

	N, N0 = params['N'], params['N0'] #parameters from data/model
	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit

	dr = 1 / float( N ) #rank increment
	p0 = N0 / float( N ) #ranking fraction

	r_vals = np.linspace( dr, p0, num=N0 ) #r values

	if ptau > 0: #for diffusion...

		if N == N0: #fix for closed systems
			r_vals[-1] = r_vals[-2] #increase by dr to avoid singularity

		levi_sea = dr * ( 1 - np.exp( -ptau ) )
		peak_sdev = np.sqrt( 2 * ptau * dr * r_vals * (1 - r_vals) )

		change_theo = 1 - np.exp( -pnu ) * ( levi_sea + np.exp( -ptau ) * st.norm.pdf( 0, scale=peak_sdev ) * dr )

		change_theo[ change_theo < 0 ] = 0 #fix for negative values

	else:
		change_theo = ( 1 - np.exp( -pnu ) ) * np.ones( N0 ) #constant change

	return change_theo


#function to compute rank out-flux in model (theo)
def flux_out_theo( params ):
	"""Compute rank out-flux in model (theo)"""

	N, N0 = params['N'], params['N0'] #parameters from data/model
	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit

	dr = 1 / float( N ) #rank increment
	p0 = N0 / float( N ) #ranking fraction

	r_vals = np.linspace( dr, p0, num=N0 ) #r values

	if ptau > 0: #for diffusion...

		if N == N0: #fix for closed systems
			r_vals[-1] = r_vals[-2] #increase by dr to avoid singularity

		peak_sdev = np.sqrt( 2 * ptau * dr * r_vals * (1 - r_vals) )
		erf_arg = ( p0 - r_vals ) / ( np.sqrt(2) * peak_sdev )
		flux_out_theo = 1 - np.exp( -pnu ) * ( 1 - (1 - p0) * ( 1 - np.exp( -ptau ) ) - 0.5 * np.exp( -ptau ) * ( 1 - ss.erf( erf_arg ) ) )

	else:
		flux_out_theo = ( 1 - np.exp( -pnu ) ) * np.ones( N0 ) #constant out-flux

	return flux_out_theo


#function to get optimal model parameters from flux/open/succ properties (all parameters at same time)
def estimate_params_all( dataname, params, saveloc, prop_dict=None, datatype='open' ):
	"""Get optimal model parameters from flux/open/succ properties (all parameters at same time)"""

	N, N0, T = params['N'], params['N0'], params['T'] #get parameters
	thres = 0.5 #threshold to calculate success/surprise measure

	#ranking fraction (float/mpm versions)
	p0_f = N0 / float( N )
	p0 = mpm.mpf(str( p0_f ))

	if prop_dict: #if we supply properties directly
		fluOprops = prop_dict['flux'][0] #mean flux over time/ranks
		openprops = prop_dict['open'][0] #average openness derivative
		success_f = prop_dict['succ'][0].loc[ thres ][ 1 ] #success time series

	else: #or from files
		param_str = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T ) #filename
		fluOprops = pd.read_pickle( saveloc + 'fluOprops_' + param_str )
		openprops = pd.read_pickle( saveloc + 'openprops_' + param_str )
		success_f = pd.read_pickle( saveloc + 'success_' + param_str ).loc[ thres ][ 1 ]

	#get flux/openness/success
	flux_f = fluOprops.mean( axis=0 ).mean()
	flux = mpm.mpf(str( flux_f ))
	open_deriv_f = openprops.loc[ 'open_deriv' ].mean()
	open_deriv = mpm.mpf(str( open_deriv_f ))
	success = mpm.mpf(str( success_f ))


	if datatype == 'open': #OPEN SYSTEMS

		if flux_f < 1:
			#ptau equation
			ptau_func = lambda pnu : pnu * ( pnu - open_deriv ) / ( p0 * open_deriv - pnu )

			#functions for pnu trascendental equation
			exp_func = lambda pnu : mpm.exp( -ptau_func( pnu ) )
			pnu_func = lambda pnu : mpm.log( ( p0 + (1 - p0) * exp_func( pnu ) ) / ( 1 - flux ) )

			#find pnu root
			pnu_lambda = lambda pnu : float( mpm.fabs( pnu - pnu_func( pnu ) ) )
			pnu_res = spo.minimize_scalar( pnu_lambda, bounds=( p0_f*open_deriv_f, open_deriv_f ), method='bounded' )
			pnu_star = pnu_res.x

			#find ptau root
			ptau_star = float( ptau_func( pnu_star ) )

		else: #exception: numerical errors
			pnu_star, ptau_star = np.nan, np.nan

	if datatype == 'closed': #CLOSED SYSTEMS

		#find roots
		ptau_star = float( mpm.log( ( 1 - thres * p0 ) / ( success - thres * p0 ) ) )
		pnu_star = 0. #closed system

	#exception: numerical errors
	if ptau_star > 1:
		ptau_star = 1.

	return flux_f, open_deriv_f, success_f, p0_f, ptau_star, pnu_star


#function to estimate system size in model that leads to number of elements ever in ranking in data
def estimate_params_size( dataname, params, loadflag, saveloc ):
	"""Estimate system size in model that leads to number of elements ever in ranking in data"""

	param_str = dataname+'.pkl' #filename for output files

	if loadflag == 'y': #load files
		params_size = np.load( saveloc + 'params_size_' + param_str )

	elif loadflag == 'n': #or else, compute sizes

		params_model = pd.read_pickle( saveloc + 'params_model_' + param_str ) #fitted parameters of dataset

		#set params for each bootstrap realisation
		params_rel = { 'N0':params['N0'], 'T':params['T'], 'ptau':params_model.loc['optimal', 'ptau'], 'pnu':params_model.loc['optimal', 'pnu'] }

		#set minimization functions
		params_func = lambda N : { **params_rel, 'N':int(np.round( N )) } #update estimated size in dict
		size_func = lambda N : np.array([ model_misc.null_model( params_func(N) )[2]['N'] for i in range(params['ntimes']) ]).mean() #get avg no. of elements ever in ranking (in simulations)
		min_func = lambda N : np.abs( size_func(N) - params['N'] ) #minimize w/ observed value in data

		#estimate system size from simulations
		size_res = spo.minimize_scalar( min_func, bounds=(params['N0'], params['N']), method='bounded', options={'disp':3} )
		params_size = int(np.round( size_res.x ))

		np.save( saveloc + 'params_size_' + param_str, params_size ) #save to file

	return params_size


#function to get optimal model parameters from mle properties (maximum likelihood estimation)
def estimate_params_MLE( dataname, params, saveloc, prop_dict=None, datatype='open', sample_frac=False ):
	"""Get optimal model parameters from mle properties (maximum likelihood estimation)"""

	N, N0, T = params['N'], params['N0'], params['T'] #get parameters

	p0 = N0 / float( N ) #ranking fraction
	dr = 1 / float( N ) #rank increment

	if prop_dict: #if we supply properties directly
		mleTprops, mleDprops = prop_dict['mle'] #MLE properties

	else: #or from files
		param_str = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T ) #filename
		mleTprops = pd.read_pickle( saveloc + 'mleTprops_' + param_str )
		mleDprops = pd.read_pickle( saveloc + 'mleDprops_' + param_str )

	#set up array of observed r, x values
	#(avoid singularity at r=1 for closed systems; pick sample for larger systems)
	rank_vals = mleDprops.columns if datatype == 'open' else mleDprops.columns[:-1]
	rx_vals = [ ( dr * ( rank + 1 ), dr * ( mleDprops.loc[time, rank] + 1 ) ) for rank, time in it.product( rank_vals, mleDprops.index ) if pd.notna( mleDprops.loc[time, rank] ) ]
	rx_vals_sample = rn.sample( rx_vals, int( sample_frac*len(rx_vals) ) ) if sample_frac else rx_vals

	#ptau MLE equations

	peak_sdev = lambda ptau, r : np.sqrt( 2 * ptau * dr * r * (1 - r) )
	gaussian = lambda ptau, r, x : st.norm.pdf( x, loc=r, scale=peak_sdev( ptau, r ) )
	numer_func = lambda ptau, r, x : 1 - ( gaussian( ptau, r, x ) / (2*ptau) )*( ( (x-r)/peak_sdev( ptau, r ) )**2 - 2*ptau*dr - 1 )
	denom_func = lambda ptau, r, x : gaussian( ptau, r, x ) + np.exp(ptau) - 1
	ptau_func = lambda ptau : np.abs( np.sum([ numer_func(ptau, r, x) / denom_func(ptau, r, x) for r, x in rx_vals_sample ]) )

	#find ptau root
	ptau_res = spo.minimize_scalar( ptau_func, bounds=( 0, 1 ), method='bounded', options={'disp':True} )
	ptau_star = ptau_res.x

#OPEN SYSTEMS
	if datatype == 'open':
		#pnu MLE equations
		surv_times_mean = mleTprops.loc['survival'].dropna().mean()
		print('1/<t>= {}'.format( 1/surv_times_mean )) #print out

		pnu_func = lambda pnu : np.abs( ( pnu**2 + 2*p0*ptau_star*pnu + p0*ptau_star**2 ) / ( pnu*( pnu + ptau_star )*( pnu + p0*ptau_star ) ) - surv_times_mean )

		#find pnu root
		pnu_res = spo.minimize_scalar( pnu_func, bounds=( 0, 1 ), method='bounded', options={'disp':True} )
		pnu_star = pnu_res.x
#CLOSED SYSTEMS
	if datatype == 'closed':
		pnu_star = 0.

	return ptau_star, pnu_star


## RUNNING MODEL MISC MODULE ##

if __name__ == "__main__":

	print( 'MODULE FOR MISCELLANEOUS FUNCTIONS FOR NULL MODEL IN FARRANKS PROJECT' )


#DEBUGGIN'

##function to run null model of rank dynamics, according to parameters
#def null_model( params ):
#	"""Run null model of rank dynamics, according to parameters"""
#
#	N0, T = params['N0'], params['T'] #parameters from data
#	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit
#
#	#initialise element time series with ranks (in ranking!)
#	#time = 0, ..., T-1, ranks = 0, ..., N0 - 1
#	elemseries = pd.DataFrame( np.empty( ( T, N0 ), dtype=int ), index=pd.Series( range(T), name='time' ), columns=pd.Series( range(N0), name='rank' ) )
#	elemseries.loc[ 0 ] = elemseries.columns #set initial condition at t=0 (using arbitrary element names)
#
#	#initialise system at time t (same as elemseries for t=0)
#	#ranks = 0, ..., N0 - 1 (0 = highest rank, N0 - 1 = lowest rank)
#	system = list( range( N0 ) ) #position = rank, value = element
#	N = N0 #initialise total number of elements in system
#
##	print( 'initial state:' )
##	print( 't 0, minit -, N0 {}, N {}, rank -, elem -, new_rank - : {}'.format( N0, N, [ R + 1 for R in system ] ) ) #to know where we are
#
# 	#rank dynamics
# 	for t in range( 1, T ): #loop through time intervals [1, ..., T-1]
# 		for minit in range( N0 ): #and N0 updates inside time intervals
#
# 			#dynamics of rank change
# 			if random.random() < ptau:
#
# 				rank = random.randint( 0, N - 1 ) #pick random rank R = 0, ..., N-1 (of whole system!)
# 				elem = system.pop( rank ) #take out element at selected rank
#
# 				new_rank = random.randint( 0, N - 1 ) #pick new rank R = 0, ..., N-1 at random (also N options!)
# 				system.insert( new_rank, elem ) #insert element at new rank
#
# #				print( 'ptau dynamics:' )
# #				print( 't {}, minit {}, N0 {}, N {}, rank {}, elem {}, new_rank {} : {}'.format( t, minit + 1, N0, N, rank + 1, elem + 1, new_rank + 1, [ R + 1 for R in system ] ) ) #to know where we are
#
# 			#dynamics of new elements
# 			if random.random() < pnu:
#
# 				elem = N #get new element (next from largest in system)
# 				rank = random.randint( 0, N ) #pick new rank R = 0, ..., N at random (1 more than system size!)
#
# 				system.insert( rank, elem ) #insert element at new rank
# 				N += 1 #increase system size
#
# #				print( 'pnu dynamics:' )
# #				print( 't {}, minit {}, N0 {}, N {}, rank {}, elem {} : {}'.format( t, minit + 1, N0, N, rank + 1, elem + 1, [ R + 1 for R in system ] ) ) #to know where we are
#
# 		#save (current) elements in ranking (R = 0, ..., N0 - 1)
# 		elemseries.loc[ t ] = system[ :N0 ]
#
# 	#initialise rank time series with the full set of elements
# 	#time = 0, ..., T-1, elements = 0, ..., N-1
# 	rankseries = pd.DataFrame( np.zeros(( T, N ))*np.nan, index=pd.Series( range(T), name='time' ), columns=pd.Series( range(N), name='element' ) )
# 	rankseries.loc[ 0, :N0-1 ] = rankseries.columns[ :N0 ] #set initial condition at t=0 (only ranking!)
#
# 	for t in range( 1, T ): #loop through time intervals [1, ..., T-1]
# 		for rank, elem in enumerate( elemseries.loc[ t ] ): #and elements in each ranking
# 			rankseries.at[ t, elem ] = rank #store current rank of element
#
# 	#drop elements that have never been in ranking!
# 	rankseries = rankseries.drop( rankseries.columns[ rankseries.apply( lambda col: col.isnull().sum() == T ) ], axis=1 )
#
# 	return rankseries, elemseries

# #function to compute rank flux in model (theo)
# def flux_time_theo( params ):
# 	"""Compute rank flux in model (theo)"""
#
# 	N, N0 = params['N'], params['N0'] #parameters from data
# 	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit
#
# 	dr = 1 / float( N ) #rankincrement
# 	p0 = N0 / float( N ) #ranking fraction
# 	s = N #MC time unit
# 	M = ptau * ( 1 - pnu * dr ) #model time scale
# 	K = ( 1 - ptau ) * pnu * ( 1 - dr ) + ( 1 - ptau ) * ( 1 - pnu ) #norm constant
#
# 	flux_time_theo = 1 - ( M * p0 * dr + K ) * ( 1 - ( M * (1 - dr) )**s ) / ( 1 - M * (1 - dr) )
#
# 	return flux_time_theo

# #function to compute system of equations for parameters in dataset (open, theo)
# def psystem_open( pvector, lower, upper, params, flux_data, open_deriv_data, min_incr ):
# 	"""Compute system of equations for parameters in dataset (open, theo)"""
#
# 	#ensure input vectors are np arrays
# 	pvector = np.asarray( pvector )
# 	lower = np.asarray( lower )
# 	upper = np.asarray( upper )
#
# 	##bring input vectors back to bounds
# 	pborder = np.where( pvector < lower, lower, pvector )
# 	pborder = np.where( pvector > upper, upper, pborder )
#
# 	#(tunable) ptau (0) and pnu (1) from (bounded) input vector
# 	params['ptau'], params['pnu'] = pborder[0], pborder[1]
#
# 	#system of equations (open)
# 	flux_root = flux_data - flux_theo( params ) #mean flux
# 	open_deriv_root = open_deriv_data - open_deriv_theo( params ) #mean openness derivative
#
# 	#function vector
# 	fborder = np.array( [ flux_root, open_deriv_root ] )
#
# 	#distance from border
# 	dist_border = np.sum( np.where( pvector < lower, lower - pvector, 0. ) ) + np.sum( np.where( pvector > upper, pvector - upper, 0. ) )
#
# 	return fborder + ( fborder + np.where( fborder > 0, min_incr, -min_incr ) ) * dist_border

# #function to estimate system size from rank openness in model (theo)
# def N_est_theo( params ):
# 	"""Estimate system size from rank openness in model (theo)"""
#
# 	N, N0, T = params['N'], params['N0'], params['T'] #parameters from data/model
# 	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit
#
# 	p0 = lambda x : N0 / float( x )
# 	exp_decay = lambda x : np.exp( -( pnu + p0(x) * ptau ) * (T-1) )
# 	inf_value = lambda x : ( pnu + ptau ) / ( pnu + p0(x) * ptau )
#
# 	N_func = lambda x : np.abs( N/x - p0(x) * ( 1 + pnu*(T-1) ) * ( exp_decay(x) + inf_value(x)*(1 - exp_decay(x)) ) )
#
# 	N_res = spo.minimize_scalar( N_func, bounds=(N0, N), method='bounded' )
# 	N_est_theo = int( N_res.x )
#
# 	# N_est_theo = int( ( ptau * N0 * N ) / ( N0 * ( pnu + ptau ) * ( 1 + pnu * (T-1) ) - pnu * ptau * N ) )
#
# 	return N_est_theo

# #function to compute rank flux through time in model (theo)
# def flux_time_theo( t, params ):
# 	"""Compute rank flux through time in model (theo)"""
#
# 	N, N0 = params['N'], params['N0'] #parameters from data/model
# 	ptau, pnu = params['ptau'], params['pnu'] #parameters to fit
#
# 	p0 = N0 / float( N ) #ranking fraction
#
# 	flux_theo = 1 - np.exp( -pnu*t ) * ( p0 + (1 - p0) * np.exp( -ptau*t ) )
#
# 	return flux_theo

# #function to get optimal model parameters from mle properties (maximum likelihood estimation)
# def estimate_params_MLE( dataname, params, saveloc, prop_dict=None, datatype='open', sample_frac=False ):
# 	"""Get optimal model parameters from mle properties (maximum likelihood estimation)"""
#
# 	N, N0, T = params['N'], params['N0'], params['T'] #get parameters
#
# 	p0 = N0 / float( N ) #ranking fraction
#
# 	if prop_dict: #if we supply properties directly
# 		mleTprops, not_used = prop_dict['mle'] #MLE properties
#
# 	else: #or from files
# 		param_str = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T ) #filename
# 		mleTprops = pd.read_pickle( saveloc + 'mleTprops_' + param_str )
#
# #OPEN SYSTEMS
# 	if datatype == 'open':
# 		#survival times mean
# 		surv_times_mean = mleTprops.loc['survival'].dropna().mean()
# 		# print('1/<t>= {}'.format( 1/surv_times_mean )) #print out
#
# 		pnu_star = 1 / surv_times_mean #solution from exp dist MLE
# #CLOSED SYSTEMS
# 	if datatype == 'closed':
# 		pnu_star = 0.
#
# #OPEN SYSTEMS
# 	if datatype == 'open':
# 		#survival times
# 		surv_times = mleTprops.loc['survival'].dropna()
# 		# print( surv_times )
#
# 		#ptau MLE equations
# 		left_func = lambda ptau : pnu_star / ( ( pnu_star + p0*ptau )*( pnu_star + ptau ) )
# 		right_func = lambda ptau : float( mpm.fsum([ t / ( 1 + p0*( mpm.exp( ptau*t ) - 1 ) ) for t in surv_times ]) / len( surv_times ) )
# 		ptau_func = lambda ptau : np.abs( left_func(ptau) - right_func(ptau) )
#
# 		#find ptau root
# 		ptau_res = spo.minimize_scalar( ptau_func, bounds=( 0, 1 ), method='bounded', options={'disp':True} )
# 		ptau_star = ptau_res.x
# #CLOSED SYSTEMS
# 	if datatype == 'closed':
# 		ptau_star = np.nan
#
# 	return ptau_star, pnu_star

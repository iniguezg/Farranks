#! /usr/bin/env python

### MODULE FOR MISCELLANEOUS FUNCTIONS FOR DATA/MODEL PROPERTIES IN FARRANKS PROJECT ###

#import modules
import numpy as np
import pandas as pd


## FUNCTIONS ##

#function to get flux properties for rank/element time series of data/model
def get_flux_props( rankseries, elemseries, params ):
	"""Get flux properties for rank/element time series of data/model"""

	N0, T = params['N0'], params['T'] #get parameters

	#initialise dataframes of flux properties (with size of ranking)
	#flux type = O (OUT of the system), I (IN to the system)
	#time = 1, ..., T-1 (without first time!), ranks = 0, ..., N0-1
	fluOprops = pd.DataFrame( np.zeros( ( T - 1, N0 ), dtype=int ), index=pd.Series( range(1, T), name='time' ), columns=pd.Series( range(N0), name='rank' ) )
	fluIprops = pd.DataFrame( np.zeros( ( T - 1, N0 ), dtype=int ), index=pd.Series( range(1, T), name='time' ), columns=pd.Series( range(N0), name='rank' ) )


	#get flux properties
	for t in range( 1, T ): #loop through time intervals [1, ..., T-1]
#		print( 't = {}'.format(t) ) #to know where we are
		for rank in range( N0 ): #loop through ranks = 0, ..., N0-1

			#OUT-flux
			elem = elemseries.at[ t-1, rank ] #select element at given rank at previous time t-1
			next_rank = rankseries.at[ t, elem ] #find next rank occupied by element at time t
			if np.isnan( next_rank ): #if element fell outside of ranking...
				fluOprops.at[ t, rank ] = 1

			#IN-flux
			elem = elemseries.at[ t, rank ] #select element at given rank at current time t
			prev_rank = rankseries.at[ t-1, elem ] #find previous rank occupied by element at time t-1
			if np.isnan( prev_rank ): #if element came from outside of ranking...
				fluIprops.at[ t, rank ] = 1

	return fluOprops, fluIprops


#function to get rank properties for rank/element time series of data/model
def get_rank_props( rankseries, elemseries, params ):
	"""Get rank properties for rank/element time series of data/model"""

	N0, T = params['N0'], params['T'] #get parameters

	#initialise dataframe of rank properties (with size of ranking)
	#property = diversity, rank change
	#ranks = 0, ..., N0-1
	rankprops = pd.DataFrame( np.zeros(( 2, N0 )), index=pd.Series( [ 'diversity', 'rankchange' ], name='property' ), columns=pd.Series( range( N0 ), name='rank' ) )

	#get rank properties
	for rank in rankprops.columns: #loop through ranks = 0, ..., N0-1

		#calculate rank diversity

		node_set = set() #initialise set of nodes for each rank
		for elem in elemseries[ rank ]: #loop through elements at given rank for all times
			node_set.add( elem ) #add node for this interval and rank

		#get diversity for rank: number of nodes in rank divided by number of time intervals
		rankprops.at[ 'diversity', rank ] = len(node_set) / float( T )

		#calculate rank change

		change_acc = 0 #initialise change accumulator in time
		for t in range( 1, T ): #loop through time intervals [1, ..., T-1]
			#if element changed between adjacent times...
			if elemseries.at[ t, rank ] != elemseries.at[ t-1, rank ]:
				change_acc += 1 #add change!

		#get average change of nodes over time, for rank
		rankprops.at[ 'rankchange', rank ] = change_acc / float( T - 1 )

	return rankprops


#function to get openness properties for rank/element time series of data/model
def get_open_props( rankseries, elemseries, params ):
	"""Get openness properties for rank/element time series of data/model"""

	N0, T = params['N0'], params['T'] #get parameters

	#initialise dataframe of openness properties
	#property = openness, openness derivative
	#time = 0, ..., T-1
	openprops = pd.DataFrame( np.zeros(( 2, T )), index=pd.Series( [ 'openness', 'open_deriv' ], name='property' ), columns=pd.Series( range(T), name='time' ) )

	all_elements = set() #initialise set of elements in system

	#get openness properties
	for t in openprops.columns: #loop through time = 0, ..., T-1

		all_elements.update( elemseries.loc[ t, : N0 - 1 ] ) #add (first N0) elements to system set
		Ntau = len( all_elements ) #number of elements in all (previous) intervals

		#get openness for this time interval
		openprops.at[ 'openness', t ] = Ntau / float( N0 )

		#and its time derivative
		if t: #for t > 0
			openprops.at[ 'open_deriv', t ] = openprops.at[ 'openness', t ] - openprops.at[ 'openness', t - 1 ]
		else: #for t = 0
			openprops.at[ 'open_deriv', t ] = np.nan #derivative undefined for first time point

	return openprops


#function to get displacement properties for rank/element time series of data/model
def get_disp_props( rankseries, elemseries, params ):
	"""Get displacement properties for rank/element time series of data/model"""

	N0, T = params['N0'], params['T'] #get parameters
	sel_ranks = [ ( (N0-1) * frac ).astype(int) for frac in np.arange(0, 1.1, 0.1) ] #selected ranks to calculate displacement

	#initialise dataframe of displacement properties
	#property = probability of displacement
	#displaced ranks = 0, ..., N0 - 1, ranks = { sel_ranks }
	dispprops = pd.DataFrame( np.zeros(( N0, len(sel_ranks) )), index=pd.Series( range( N0 ), name='disp_rank' ), columns=pd.Series( sel_ranks, name='rank' ) )

	#get displacement properties
	for rank in sel_ranks: #loop through selected ranks
		for t in range( 1, T ): #loop through time intervals [1, ..., T-1]

			elem = elemseries.at[ t-1, rank ] #get element in rank at previous time
			disp_rank = rankseries.at[ t, elem ] #get displaced rank of that element one time later

			if np.isnan( disp_rank ) == False: #if we're still in ranking
				dispprops.at[ int( disp_rank ), rank ] += 1 #add count to probability distribution

		dispprops[ rank ] /= float( T - 1 ) #normalise distribution (counting nans!)

	return dispprops


#function to get displacement-in-time properties for rank/element time series of data/model
def get_disp_time_props( rankseries, elemseries, params ):
	"""Get displacement-in-time properties for rank/element time series of data/model"""

	N0, T = params['N0'], params['T'] #get parameters
	sel_rank = params['sel_rank'] #selected rank = 0, ..., N0-1 to calculate displacement
	lags = range(T) #possible time lags [0, ..., T-1]

	#initialise dataframe of displacement-in-time properties
	#property = probability of displacement
	#displaced ranks = 0, ..., N0 - 1, lags = [ lags ]
	dispprops = pd.DataFrame( np.zeros(( N0, len(lags) )), index=pd.Series( range( N0 ), name='disp_rank' ), columns=pd.Series( lags, name='lag' ) )

	for lag in lags: #loop through lags [0, ..., T-1]
		for t_ini in range( T - lag ): #loop through initial times [0, ..., T - lag - 1]
			t_fin = t_ini + lag #set final (lagged) time

			elem = elemseries.at[ t_ini, sel_rank ] #get element in rank at initial time
			disp_rank = rankseries.at[ t_fin, elem ] #get displaced rank of that element one lag later

			if np.isnan( disp_rank ) == False: #if we're still in ranking
				dispprops.at[ int( disp_rank ), lag ] += 1 #add count to probability distribution

		dispprops[ lag ] /= float( T - lag ) #normalise distribution (counting nans!)

	return dispprops


#function to get success properties for rank/element time series of data/model
def get_succ_props( rankseries, elemseries, params ):
	"""Get success properties for rank/element time series of data/model"""

	N0, T = params['N0'], params['T'] #get parameters
	lags = range(T) #possible time lags [0, ..., T-1]
#	thresholds = np.arange(0.1, 1, 0.1) #thresholds to define top/bottom of ranking
	thresholds = [ 0.5 ]

	#initialise dataframes of success/surprise time series
	#[ thresholds ], [ lags ]
	success = pd.DataFrame( np.zeros( ( len(thresholds), len(lags) ) ), index=pd.Series( thresholds, name='threshold' ), columns=pd.Series( lags, name='lag' ) )
	surprise = pd.DataFrame( np.zeros( ( len(thresholds), len(lags) ) ), index=pd.Series( thresholds, name='threshold' ), columns=pd.Series( lags, name='lag' ) )

	#get success/surprise
	for thres in thresholds: #loop through thresholds
#		print( 'threshold = {:.1f}'.format(thres) ) #to know where we are

		thres_rank = int( thres * N0 ) #get threshold rank

		for lag in lags: #loop through lags [0, ..., T-1]
			success_buffer, success_count = 0, 0 #initialise buffers/counters
			surprise_buffer, surprise_count = 0, 0

			for t_ini in range( T - lag ): #loop through initial times [0, ..., T - lag - 1]
				t_fin = t_ini + lag #set final (lagged) time

				#get sets of elements at top/bottom of ranking at the initial/final time
				set_top_ini = set( elemseries.loc[ t_ini, 0 : thres_rank ] )
				set_top_fin = set( elemseries.loc[ t_fin, 0 : thres_rank ] )
				set_bot_ini = set( elemseries.loc[ t_ini, thres_rank+1 : N0-1 ] )

				#fill out buffer/counter for success (permanence on top ranking)
				success_buffer += len( set_top_ini.intersection( set_top_fin ) )
				success_count += len( set_top_ini )

				#fill out buffer/counter for surprise (rise from bottom to top ranking)
				surprise_buffer += len( set_bot_ini.intersection( set_top_fin ) )
				surprise_count += len( set_bot_ini )

			#get success/surprise for given lag and store in dataframe
			success.at[ thres, lag ] = success_buffer / float( success_count )
			surprise.at[ thres, lag ] = surprise_buffer / float( surprise_count )

	return success, surprise


#function to get MLE properties for rank/element time series of data/model
def get_MLE_props( rankseries, elemseries, params ):
	"""Get MLE properties for rank/element time series of data/model"""

	N, N0, T = params['N'], params['N0'], params['T'] #parameters from data

	#initialise dataframes of MLE properties
	#property = survival time
	#elements = N (sorted) names in data
	mleTprops = pd.DataFrame( np.zeros(( 1, N ))*np.nan, index=pd.Series( [ 'survival' ], name='property' ), columns=rankseries.columns )
	#property = rank displacement
	#time = 1, ..., T-1 (without first time!), ranks = 0, ..., N0-1
	mleDprops = pd.DataFrame( np.zeros(( T - 1, N0 ))*np.nan, index=pd.Series( range(1, T), name='time' ), columns=pd.Series( range(N0), name='rank' ) )

	#survival time
	for elem in rankseries: #loop through elements in system
		#get first/last time element was in ranking
		rankseries_elem = rankseries[ elem ][ rankseries[ elem ] < N0 ]
		t_ini = rankseries_elem.index.min()
		t_fin = rankseries_elem.index.max()

		#get survival time (last time inclusive!)
		#only actual survival times! (t_fin < T-1), not lower bounds
		if t_fin < T-1:
			mleTprops.at[ 'survival', elem ] = t_fin - t_ini + 1

	#rank displacement
	for rank in range( N0 ): #loop through ranks = 0, ..., N0-1
		for t in range( 1, T ): #loop through time intervals [1, ..., T-1]

			elem = elemseries.at[ t-1, rank ] #get element in rank at previous time
			disp_rank = rankseries.at[ t, elem ] #get displaced rank of that element one time later

			if np.isnan( disp_rank ) == False: #if we're still in ranking
				mleDprops.at[ t, rank ] = disp_rank

	return mleTprops, mleDprops


#function to get transition properties for rank/element time series of data/model
def get_tran_props( rankseries, elemseries, params ):
	"""Get transition properties for rank/element time series of data/model"""

	N0, T = params['N0'], params['T'] #get parameters
	thres = params['thres'] #transition threshold

	thres_rank = int( thres * N0 ) #get threshold rank
	#transition types (between top/center/bottom of ranking defined by threshold)
	tran_types = [ 'top2top', 'top2cen', 'top2bot',
				   'cen2top', 'cen2cen', 'cen2bot',
				   'bot2top', 'bot2cen', 'bot2bot' ]
	lags = range( T ) #possible time lags [0, ..., T-1]

	#initialise dataframes of transition time series
	#[ transition type ], [ lags ]
	tranprops = pd.DataFrame( np.zeros( ( len(tran_types), len(lags) ) ), index=pd.Series( tran_types, name='tran_type' ), columns=pd.Series( lags, name='lag' ) )

	for lag in lags: #loop through lags [0, ..., T-1]
		buffer = { tran_type : 0 for tran_type in tran_types } #initialise buffers/counters
		count = { tran_type : 0 for tran_type in tran_types }

		for t_ini in range( T - lag ): #loop through initial times [0, ..., T - lag - 1]
			t_fin = t_ini + lag #set final (lagged) time

			#get sets of elements at top/center/bottom of ranking at the initial/final time
			top_ini = elemseries.loc[ t_ini, 0 : thres_rank - 1 ]
			top_fin = elemseries.loc[ t_fin, 0 : thres_rank - 1 ]
			cen_ini = elemseries.loc[ t_ini, thres_rank : ( N0 - thres_rank ) - 1 ]
			cen_fin = elemseries.loc[ t_fin, thres_rank : ( N0 - thres_rank ) - 1 ]
			bot_ini = elemseries.loc[ t_ini, ( N0 - thres_rank ) : N0 - 1 ]
			bot_fin = elemseries.loc[ t_fin, ( N0 - thres_rank ) : N0 - 1 ]

			#fill out buffers/counters for all transitions

			buffer[ 'top2top' ] = top_ini.isin( top_fin ).sum()
			count[  'top2top' ] = len( top_ini )
			buffer[ 'top2cen' ] = top_ini.isin( cen_fin ).sum()
			count[  'top2cen' ] = len( top_ini )
			buffer[ 'top2bot' ] = top_ini.isin( bot_fin ).sum()
			count[  'top2bot' ] = len( top_ini )

			buffer[ 'cen2top' ] = cen_ini.isin( top_fin ).sum()
			count[  'cen2top' ] = len( cen_ini )
			buffer[ 'cen2cen' ] = cen_ini.isin( cen_fin ).sum()
			count[  'cen2cen' ] = len( cen_ini )
			buffer[ 'cen2bot' ] = cen_ini.isin( bot_fin ).sum()
			count[  'cen2bot' ] = len( cen_ini )

			buffer[ 'bot2top' ] = bot_ini.isin( top_fin ).sum()
			count[  'bot2top' ] = len( bot_ini )
			buffer[ 'bot2cen' ] = bot_ini.isin( cen_fin ).sum()
			count[  'bot2cen' ] = len( bot_ini )
			buffer[ 'bot2bot' ] = bot_ini.isin( bot_fin ).sum()
			count[  'bot2bot' ] = len( bot_ini )

		#average transitions
		for tran_type in tran_types: #loop through transition types
			if count[ tran_type ]: #if we have anything...
				tranprops.at[ tran_type, lag ] = buffer[ tran_type ] / float( count[ tran_type ] )
			else:
				tranprops.at[ tran_type, lag ] = np.nan

	return tranprops


## RUNNING PROPS MISC MODULE ##

if __name__ == "__main__":

	print( 'MODULE FOR MISCELLANEOUS FUNCTIONS FOR DATA/MODEL PROPERTIES IN FARRANKS PROJECT' )

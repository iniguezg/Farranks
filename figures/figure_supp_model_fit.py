#! /usr/bin/env python

### SCRIPT FOR PLOTTING SUPP FIGURE (THEO MODEL FIT) IN FARRANKS PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as skm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc, model_misc


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#fitting variables
#	gof_str = 'MSE' #Goodness-of-fit measure. Default: Mean Squared Error (MSE)

	#rank measure parameters
	thres = 0.5 #threshold to calculate success/surprise measure

	#flag and locations
	loadflag = 'y'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_data = root_loc+'nullModel/v4/files/' #location of output files

	#dataset short names, types, and colors

#	datasets = { 'AcademicRanking' : 'universities', 'AtlasComplex' : 'countries', 'Citations' : 'scientists', 'Cities_RU' : 'cities (RU)', 'Cities_UK' : 'cities (GB)', 'Earthquakes_avgMagnitude' : 'regions JP (quake mag)', 'Earthquakes_numberQuakes' : 'regions JP (quakes)', 'english' : 'English', 'enron-sent-mails-weekly' : 'Enron emails', 'FIDEFemale' : 'chess players (female)', 'FIDEMale' : 'chess players (male)', 'Football_FIFA' : 'national football teams', 'Football_Scorers' : 'Football scorers', 'Fortune' : 'companies', 'french' : 'French', 'german' : 'German', 'github-watch-weekly' : 'GitHub repositories', 'Golf_OWGR' : 'golf players', 'Hienas' : 'hyenas', 'italian' : 'Italian', 'metroMex' : 'metro stations (Mexico)', 'Nascar_BuschGrandNational' : 'Nascar drivers (Busch)', 'Nascar_WinstonCupGrandNational' : 'Nascar drivers (Winston Cup)', 'Poker_GPI' : 'poker players', 'russian' : 'Russian', 'spanish' : 'Spanish', 'Tennis_ATP' : 'tennis players', 'TheGuardian_avgRecommends' : 'The Guardian readers (recc)', 'TheGuardian_numberComments' : 'The Guardian readers (comm)', 'UndergroundByWeek' : 'metro stations (London)' } #name dict
	datasets = { 'AcademicRanking' : 'universities', 'AtlasComplex' : 'countries', 'Citations' : 'scientists', 'Cities_RU' : 'cities (RU)', 'Cities_UK' : 'cities (GB)', 'Earthquakes_avgMagnitude' : 'regions JP (quake mag)', 'Earthquakes_numberQuakes' : 'regions JP (quakes)', 'english' : 'English', 'enron-sent-mails-weekly' : 'Enron emails', 'FIDEFemale' : 'chess players (female)', 'FIDEMale' : 'chess players (male)', 'Football_FIFA' : 'national football teams', 'Football_Scorers' : 'Football scorers', 'Fortune' : 'companies', 'french' : 'French', 'german' : 'German', 'github-watch-weekly' : 'GitHub repositories', 'Golf_OWGR' : 'golf players', 'Hienas' : 'hyenas', 'metroMex' : 'metro stations (Mexico)', 'Nascar_BuschGrandNational' : 'Nascar drivers (Busch)', 'Nascar_WinstonCupGrandNational' : 'Nascar drivers (Winston Cup)', 'Poker_GPI' : 'poker players', 'Tennis_ATP' : 'tennis players', 'TheGuardian_avgRecommends' : 'The Guardian readers (recc)', 'TheGuardian_numberComments' : 'The Guardian readers (comm)', 'UndergroundByWeek' : 'metro stations (London)' } #name dict
#	datasets = { 'VideogameEarnings' : 'videogame\nplayers', 'Virus' : 'viruses' } #shady data

	datatypes = { 'AcademicRanking' : 'open', 'AtlasComplex' : 'open', 'Citations' : 'open', 'Cities_RU' : 'open', 'Cities_UK' : 'closed', 'Earthquakes_avgMagnitude' : 'closed', 'Earthquakes_numberQuakes' : 'closed', 'english' : 'open', 'enron-sent-mails-weekly' : 'open', 'FIDEFemale' : 'open', 'FIDEMale' : 'open', 'Football_FIFA' : 'closed', 'Football_Scorers' : 'open', 'Fortune' : 'open', 'french' : 'open', 'german' : 'open', 'github-watch-weekly' : 'open', 'Golf_OWGR' : 'open', 'Hienas' : 'open', 'italian' : 'open', 'metroMex' : 'closed', 'Nascar_BuschGrandNational' : 'open', 'Nascar_WinstonCupGrandNational' : 'open', 'Poker_GPI' : 'open', 'russian' : 'open', 'spanish' : 'open','Tennis_ATP' : 'open', 'TheGuardian_avgRecommends' : 'open', 'TheGuardian_numberComments' : 'open', 'UndergroundByWeek' : 'closed' } #type dict
#	datasets = { 'VideogameEarnings' : 'open', 'Virus' : 'open' } #shady data

	datacats = { 'AcademicRanking' : 'society', 'AtlasComplex' : 'economics', 'Citations' : 'society', 'Cities_RU' : 'infrastructure', 'Cities_UK' : 'infrastructure', 'Earthquakes_avgMagnitude' : 'nature', 'Earthquakes_numberQuakes' : 'nature', 'english' : 'languages', 'enron-sent-mails-weekly' : 'society', 'FIDEFemale' : 'sports', 'FIDEMale' : 'sports', 'Football_FIFA' : 'sports', 'Football_Scorers' : 'sports', 'Fortune' : 'economics', 'french' : 'languages', 'german' : 'languages', 'github-watch-weekly' : 'society', 'Golf_OWGR' : 'sports', 'Hienas' : 'nature', 'italian' : 'languages', 'metroMex' : 'infrastructure', 'Nascar_BuschGrandNational' : 'sports', 'Nascar_WinstonCupGrandNational' : 'sports', 'Poker_GPI' : 'sports', 'russian' : 'languages', 'spanish' : 'languages','Tennis_ATP' : 'sports', 'TheGuardian_avgRecommends' : 'society', 'TheGuardian_numberComments' : 'society', 'UndergroundByWeek' : 'infrastructure' } #type dict
#	datasets = { 'VideogameEarnings' : 'economics', 'Virus' : 'nature' } #shady data

	palette = sns.color_palette( 'Pastel1', n_colors=7 ) #selected colormap for types
	datacols = { 'society' : palette[0], 'economics' : palette[1], 'nature' : palette[2], 'infrastructure' : palette[3], 'sports' : palette[4], 'languages' : palette[6] } #set color for dataset type

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 13,
	'text_size' : 10,
	'marker_size' : 3,
	'linewidth' : 3,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':10 },
	'legend_hlen' : 1.8,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }


	#get parameters for all datasets
	params_data = pd.read_pickle( saveloc_data+'params_data.pkl' )

	for dataname in datasets: #loop through considered datasets
#	for dataname in [ 'AcademicRanking' ]:

		print( 'dataset name: ' + dataname ) #print dataset

		#model fit parameters
		datatype = datatypes[ dataname ] #dataset type: open, closed

		#properties to fit
		if datatype == 'open':
			prop_names = { 'flux_time' : r'$F_t$',
					   'openness' : r'$o_t$',
					   'flux_rank' : r'$F^{\pm}_R$',
					   'change' : r'$C_R$',
					   'success' : r'$S^{++}_l$' }
		elif datatype == 'closed':
			prop_names = { 'change' : r'$C_R$',
					   'success' : r'$S^{++}_l$' }

		#plot variables
		fig_props = { 'fig_num' : 1,
		'fig_size' : (10, 6),
		'aspect_ratio' : (2, 3),
		'grid_params' : dict( left=0.065, bottom=0.09, right=0.99, top=0.87, wspace=0.3, hspace=0.5 ),
		'dpi' : 300,
		'savename' : 'figure_supp_model_fit_'+dataname }


		## DATA AND MODEL ##

		#get parameters for dataset
		params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

		#get model parameters for selected dataset
#		params_model = data_misc.data_estimate_params_theo( dataname, params, loadflag, saveloc_data, gof_str=gof_str, datatype=datatype )
		params_model = data_misc.data_estimate_params_all( dataname, params, loadflag, saveloc_data, datatype=datatype )
		# params_model = data_misc.data_estimate_params_MLE( dataname, params, loadflag, saveloc_data, datatype=datatype )

		## PLOTTING ##

		#initialise plot
		sns.set( style="white" ) #set fancy fancy plot
		#convert fig size to inches
		fig = plt.figure( fig_props['fig_num'], figsize=np.array( fig_props['fig_size'] ) )
		plt.clf()
		grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
		grid.update( **fig_props['grid_params'] )

		grid_pos = -1
		for prop_name, prop_label in prop_names.items(): #loop through measures to fit
			grid_pos += 1

			#initialise subplot
			ax = plt.subplot( grid[ grid_pos ] )
			sns.despine( ax=ax ) #take out spines
			if prop_name == 'flux_time':
				plt.xlabel( '$t / T$', size=plot_props['xylabel'], labelpad=2 )
			if prop_name == 'openness':
				plt.xlabel( '$t / T$', size=plot_props['xylabel'], labelpad=2 )
			if prop_name == 'flux_rank':
				plt.xlabel( '$R / N_0$', size=plot_props['xylabel'], labelpad=2 )
			if prop_name == 'change':
				plt.xlabel( '$R / N_0$', size=plot_props['xylabel'], labelpad=2 )
			if prop_name == 'success':
				plt.xlabel( '$l / T$', size=plot_props['xylabel'], labelpad=2 )
			plt.ylabel( prop_label, size=plot_props['xylabel'], labelpad=2 )

			#prepare data/model

			#set parameters per rank property
#			params[ 'pnu' ], params[ 'ptau' ] = params_model.loc[ prop_name, [ 'pnu', 'ptau' ] ]
			params[ 'pnu' ], params[ 'ptau' ] = params_model.loc[ 'optimal', [ 'pnu', 'ptau' ] ]

			if prop_name == 'flux_time':
				xplot = np.arange( 1, T ) / T #normalised time = ( 1, ..., T-1 ) / T

				fluOprops_data = pd.read_pickle( saveloc_data + 'fluOprops_' + param_str_data )
				yplot_data = fluOprops_data.mean( axis=1 ) #OUT-flow (=IN-flow when averaged over Nt)

				yplot_model = model_misc.flux_theo( params ) * np.ones(( T - 1 )) #constant flow over time

			if prop_name == 'openness':
				xplot = np.arange( T ) / T  #normalised time = ( 0, ..., T-1 ) / T

				openprops_data = pd.read_pickle( saveloc_data + 'openprops_' + param_str_data )
				openness_data = openprops_data.loc[ 'openness' ]
				yplot_data = openness_data / openness_data.max() #normalise by max openness!

				openness_model = model_misc.openness_theo( params )
				yplot_model = openness_model / openness_model.max() #normalise by max openness!

			if prop_name == 'flux_rank':
				xplot = np.arange( 1, N0+1 ) / N0 #normalised rank = ( 1, ..., N0 ) / N0

				fluOprops_data = pd.read_pickle( saveloc_data + 'fluOprops_' + param_str_data )
				yplot_data = fluOprops_data.mean( axis=0 ) #OUT-flow (averaged over time t)

				yplot_model = model_misc.flux_out_theo( params )

			if prop_name == 'change':
				xplot = np.arange( 1, N0+1 ) / N0 #normalised rank = ( 1, ..., N0 ) / N0

				rankprops_data = pd.read_pickle( saveloc_data + 'rankprops_' + param_str_data )
				yplot_data = rankprops_data.loc[ 'rankchange' ]

				yplot_model = model_misc.change_theo( params )

			if prop_name == 'success':
				xplot = np.arange( T ) / T #normalised time = ( 0, ..., T-1 ) / T

				success_data = pd.read_pickle( saveloc_data + 'success_' + param_str_data )
				yplot_data = success_data.loc[ thres ]

				yplot_model = model_misc.success_theo( thres, params )

			#plot plot!
			plt.plot( xplot, yplot_data, label='data', c=datacols[ datacats[ dataname ] ], lw=plot_props['linewidth'], rasterized=False )
			plt.plot( xplot, yplot_model, label='model', c='0.5', lw=plot_props['linewidth']-1, rasterized=False )

			#GoF test
			MSE = skm.mean_squared_error( yplot_data, yplot_model )

			#texts

			if grid_pos == 1:
				plt.text( 0.5, 1.35, datasets[ dataname ], va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )
				plt.text( 0.5, 1.23, '$N = $ {}, $N_0 = $ {}, $T = $ {}'.format( N, N0, T ), va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

			text_str = r'$p_{\tau} =$ '+'{:.3f}'.format(params['ptau'])+r', $p_{\nu} =$ '+'{:.3f}'.format(params['pnu'])
			plt.text( 0.5, 1.05, text_str, va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

			gof_str = 'MSE $=$ {:.3f}'.format( MSE )
			plt.text( 0.5, 0.95, gof_str, va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )


			#finalise subplot
			plt.axis([ -0.05, 1.05, -0.05, 1.05 ])
			ax.locator_params( nbins=4 )
			ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )

			#legend
			if grid_pos == 2:
				leg = plt.legend( loc='upper right', bbox_to_anchor=( 1.04, 1.43 ), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )

		#finalise plot
		if fig_props['savename'] != '':
			plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
	#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )


#DEBUGGIN'

			# if datatype == 'open':
			# 	params[ 'pnu' ], params[ 'ptau' ] = params_model.loc[ 'flux_time', [ 'pnu', 'ptau' ] ]
			# elif datatype == 'closed':
			# 	params[ 'pnu' ], params[ 'ptau' ] = params_model.loc[ 'success', [ 'pnu', 'ptau' ] ]

#! /usr/bin/env python

### SCRIPT FOR PLOTTING SUPP FIGURE (THEO MODEL PHASE DIAGRAM) IN FARRANKS PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#fitting variables
	gof_str = 'MSE' #Goodness-of-fit measure. Default: Mean Squared Error (MSE)

	par_step = 0.001 #parameter step
	ptau_vals = np.arange( 0, 1 + par_step/2, par_step ) #diffusion probability
	pnu_vals = np.arange( 0, 1 + par_step/2, par_step ) #replacement probability

	#flag and locations
	loadflag = 'y'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_data = root_loc+'nullModel/v4/files/' #location of output files

	#dataset short names, types, and colors

	datasets = { 'AcademicRanking' : 'universities', 'AtlasComplex' : 'countries', 'Citations' : 'scientists', 'Cities_RU' : 'cities (RU)', 'Cities_UK' : 'cities (GB)', 'Earthquakes_avgMagnitude' : 'regions JP (quake mag)', 'Earthquakes_numberQuakes' : 'regions JP (quakes)', 'english' : 'English', 'enron-sent-mails-weekly' : 'Enron emails', 'FIDEFemale' : 'chess players (female)', 'FIDEMale' : 'chess players (male)', 'Football_FIFA' : 'national football teams', 'Football_Scorers' : 'Football scorers', 'Fortune' : 'companies', 'french' : 'French', 'german' : 'German', 'github-watch-weekly' : 'GitHub repositories', 'Golf_OWGR' : 'golf players', 'Hienas' : 'hyenas', 'italian' : 'Italian', 'metroMex' : 'metro stations (Mexico)', 'Nascar_BuschGrandNational' : 'Nascar drivers (Busch)', 'Nascar_WinstonCupGrandNational' : 'Nascar drivers (Winston Cup)', 'Poker_GPI' : 'poker players', 'russian' : 'Russian', 'spanish' : 'Spanish', 'Tennis_ATP' : 'tennis players', 'TheGuardian_avgRecommends' : 'The Guardian readers (recc)', 'TheGuardian_numberComments' : 'The Guardian readers (comm)', 'UndergroundByWeek' : 'metro stations (London)' } #name dict
#	datasets = { 'VideogameEarnings' : 'videogame\nplayers', 'Virus' : 'viruses' } #shady data

	datatypes = { 'AcademicRanking' : 'open', 'AtlasComplex' : 'open', 'Citations' : 'open', 'Cities_RU' : 'open', 'Cities_UK' : 'closed', 'Earthquakes_avgMagnitude' : 'closed', 'Earthquakes_numberQuakes' : 'closed', 'english' : 'open', 'enron-sent-mails-weekly' : 'open', 'FIDEFemale' : 'open', 'FIDEMale' : 'open', 'Football_FIFA' : 'open', 'Football_Scorers' : 'open', 'Fortune' : 'open', 'french' : 'open', 'german' : 'open', 'github-watch-weekly' : 'open', 'Golf_OWGR' : 'open', 'Hienas' : 'open', 'italian' : 'open', 'metroMex' : 'closed', 'Nascar_BuschGrandNational' : 'open', 'Nascar_WinstonCupGrandNational' : 'open', 'Poker_GPI' : 'open', 'russian' : 'open', 'spanish' : 'open','Tennis_ATP' : 'open', 'TheGuardian_avgRecommends' : 'open', 'TheGuardian_numberComments' : 'open', 'UndergroundByWeek' : 'closed' } #type dict
#	datasets = { 'VideogameEarnings' : 'open', 'Virus' : 'open' } #shady data

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 13,
	'text_size' : 10,
	'marker_size' : 5,
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
		'grid_params' : dict( left=0.05, bottom=0.085, right=0.975, top=0.87, wspace=0.3, hspace=0.5 ),
		'dpi' : 300,
		'savename' : 'figure_supp_model_PD_'+dataname }


		## DATA AND MODEL ##

		#get parameters for dataset
		params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data

		#get model phase diagram and optimal parameters for given dataset

		model_PD = data_misc.data_estimate_PD_theo( dataname, params, loadflag, saveloc_data, par_step=par_step, gof_str=gof_str, datatype=datatype )

		params_model = data_misc.data_estimate_params_theo( dataname, params, loadflag, saveloc_data, gof_str=gof_str, datatype=datatype )


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

			#plot plot!
			zplot = np.log10( model_PD[ prop_name ] ) #logarithm of error measure
			sns.heatmap( zplot, ax=ax, cmap='GnBu', xticklabels=20, yticklabels=20 )

			xplot = params_model.loc[ prop_name, 'ptau' ] * ( len( ptau_vals ) - 1 )
			yplot = ( 1 - params_model.loc[ prop_name, 'pnu' ] ) * len( pnu_vals )
			plt.plot( xplot, yplot, 'or', ms=plot_props['marker_size'] )

			#texts

			if grid_pos == 1:
				plt.text( 0.5, 1.35, datasets[ dataname ], va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )
				plt.text( 0.5, 1.2, '$N = $ {}, $N_0 = $ {}, $T = $ {}'.format( N, N0, T ), va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

			#name of rank measure
			plt.text( 0.5, 1.05, prop_label, va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

			#extra texts
			cbar_str = r'$\log_{10}$ ' + gof_str
			plt.text( 1.2, 1.05, cbar_str, va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )


		#finalise plot
		if fig_props['savename'] != '':
			plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
	#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )

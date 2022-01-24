#! /usr/bin/env python

#Farranks - Exploring rank dynamics in complex systems
#Copyright (C) 2022 Gerardo Iñiguez

### SCRIPT FOR PLOTTING SUPP FIGURE (MODEL PHASE DIAGRAM) IN FARRANKS PROJECT ###

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

	#model fit parameters
	dataname = 'TheGuardian_numberComments' #dataset
	datatype = 'open' #dataset type: open, closed
	ntimes = 10 #number of model realisations
	ptau_vals = np.arange( 0, 1.01, 0.02 ) #explored parameters
	pnu_vals = np.arange( 0, 1.01, 0.02 )
#	pnu_vals = np.array([ 0. ])
	prop_names = { 'flux_time' : r'$F_t$',
				   'openness' : r'$o_t$',
				   'flux_rank' : r'$F^{\pm}_R$',
				   'change' : r'$C_R$',
				   'diversity' : r'$d_R / d_m$',
				   'success' : r'$S^{++}_l$' } #properties to fit

	#flag and locations
	loadflag = 'y'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_data = root_loc+'nullModel/v4/files/' #location of output files
	saveloc_model = root_loc+'nullModel/v4/files/model/'

	#dataset short names, types, and colors

	datasets = { 'AcademicRanking' : 'universities', 'AtlasComplex' : 'countries', 'Citations' : 'scientists', 'Cities_RU' : 'cities (RU)', 'Cities_UK' : 'cities (GB)', 'Earthquakes_avgMagnitude' : 'regions JP (quake mag)', 'Earthquakes_numberQuakes' : 'regions JP (quakes)', 'english' : 'English', 'enron-sent-mails-weekly' : 'Enron emails', 'FIDEFemale' : 'chess players (female)', 'FIDEMale' : 'chess players (male)', 'Football_FIFA' : 'national football teams', 'Football_Scorers' : 'Football scorers', 'Fortune' : 'companies', 'french' : 'French', 'german' : 'German', 'github-watch-weekly' : 'GitHub repositories', 'Golf_OWGR' : 'golf players', 'Hienas' : 'hyenas', 'italian' : 'Italian', 'metroMex' : 'metro stations (Mexico)', 'Nascar_BuschGrandNational' : 'Nascar drivers (Busch)', 'Nascar_WinstonCupGrandNational' : 'Nascar drivers (Winston Cup)', 'Poker_GPI' : 'poker players', 'russian' : 'Russian', 'spanish' : 'Spanish', 'Tennis_ATP' : 'tennis players', 'TheGuardian_avgRecommends' : 'The Guardian readers (recc)', 'TheGuardian_numberComments' : 'The Guardian readers (comm)', 'UndergroundByWeek' : 'metro stations (London)' } #name dict

	datatypes = { 'AcademicRanking' : 'society', 'AtlasComplex' : 'economics', 'Citations' : 'society', 'Cities_RU' : 'infrastructure', 'Cities_UK' : 'infrastructure', 'Earthquakes_avgMagnitude' : 'nature', 'Earthquakes_numberQuakes' : 'nature', 'english' : 'languages', 'enron-sent-mails-weekly' : 'society', 'FIDEFemale' : 'sports', 'FIDEMale' : 'sports', 'Football_FIFA' : 'sports', 'Football_Scorers' : 'sports', 'Fortune' : 'economics', 'french' : 'languages', 'german' : 'languages', 'github-watch-weekly' : 'society', 'Golf_OWGR' : 'sports', 'Hienas' : 'nature', 'italian' : 'languages', 'metroMex' : 'infrastructure', 'Nascar_BuschGrandNational' : 'sports', 'Nascar_WinstonCupGrandNational' : 'sports', 'Poker_GPI' : 'sports', 'russian' : 'languages', 'spanish' : 'languages','Tennis_ATP' : 'sports', 'TheGuardian_avgRecommends' : 'society', 'TheGuardian_numberComments' : 'society', 'UndergroundByWeek' : 'infrastructure' } #type dict

	palette = sns.color_palette( 'Pastel1', n_colors=7 ) #selected colormap for types
	datacols = { 'society' : palette[0], 'economics' : palette[1], 'nature' : palette[2], 'infrastructure' : palette[3], 'sports' : palette[4], 'languages' : palette[6] } #set color for dataset type

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

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (10, 6),
	'aspect_ratio' : (2, 3),
#	'grid_params' : dict( left=0.05, bottom=0.085, right=0.975, top=0.87, wspace=0.3, hspace=0.5 ),
	'grid_params' : dict( left=0.07, bottom=0.085, right=0.985, top=0.87, wspace=0.3, hspace=0.5 ),
	'dpi' : 300,
#	'savename' : 'figure_supp_model_PD_'+dataname }
	'savename' : 'figure_supp_model_PD_'+dataname+'_diagonal' }


	## DATA AND MODEL ##

	#get parameters for all datasets
	params_data = pd.read_pickle( saveloc_data+'params_data.pkl' )

	#get parameters for dataset
	params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)
	N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data

	params[ 'ntimes' ] = ntimes #set number of realisations

	#get model phase diagram and optimal parameters for given dataset
	model_PD, params_model = data_misc.data_estimate_params( dataname, params, loadflag, saveloc_data, saveloc_model )


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
		if datatype == 'open':
#			zplot = np.log10( model_PD[ prop_name ] ) #logarithm of error measure
#			sns.heatmap( zplot, ax=ax, cmap='GnBu', xticklabels=10, yticklabels=10 )
#
#			xplot = params_model.loc[ prop_name, 'ptau' ] * ( len( ptau_vals ) - 1 )
#			yplot = ( 1 - params_model.loc[ prop_name, 'pnu' ] ) * len( pnu_vals )
#			plt.plot( xplot, yplot, 'or', ms=plot_props['marker_size'] )

			#plot plot!
			xplot = ptau_vals
			yplot = [ model_PD[ prop_name ].loc[ ptau, ptau ] for ptau in ptau_vals ] #go along diagonal!
			plt.semilogy( xplot, yplot, c=datacols[ datatypes[ dataname ] ], lw=plot_props['linewidth'], rasterized=False, zorder=1 )

			#info about minimum ptau
			xplot_min = xplot[ np.nanargmin( yplot ) ]
			plt.axvline( xplot_min, ls='--', c='0.8', lw=2, zorder=0 )
			plt.text( xplot_min+0.1, 0.99, r'$p_{\tau}=$ ' + '{:.2f}'.format( xplot_min ), va='top', ha='left', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		elif datatype == 'closed':
			#initialise subplot
			if grid_pos in [0, 3]:
				plt.ylabel( 'MSE', size=plot_props['xylabel'], labelpad=2 )

			#plot plot!
			logflag = False if prop_name in [ 'flux_time', 'openness', 'flux_rank' ] else True
			yplot = model_PD[ prop_name ].loc[ 0., : ]
			yplot.plot( logy=logflag, c=datacols[ datatypes[ dataname ] ], lw=plot_props['linewidth'], rasterized=False )


		#texts

		if grid_pos == 1:
			plt.text( 0.5, 1.35, datasets[ dataname ], va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )
			plt.text( 0.5, 1.2, '$N = $ {}, $N_0 = $ {}, $T = $ {}'.format( N, N0, T ), va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#name of rank measure
		plt.text( 0.5, 1.05, prop_label, va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#extra texts
		if datatype == 'open':
#			cbar_str = '$\log_{10}$ MSE'
#			plt.text( 1.2, 1.05, cbar_str, va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )
			plt.xlabel( r'$p_{\tau}$', size=plot_props['xylabel'], labelpad=2 )
			plt.ylabel( 'MSE', size=plot_props['xylabel'], labelpad=2 )


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )

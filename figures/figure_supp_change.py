#! /usr/bin/env python

#Farranks - Exploring rank dynamics in complex systems
#Copyright (C) 2022 Gerardo Iñiguez

### SCRIPT FOR PLOTTING SUPP FIGURE (CHANGE) IN FARRANKS PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc, model_misc


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##
	prop_name = 'change' #rank property to consider

	#flag and locations
	loadflag = 'y'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_data = root_loc+'nullModel/v4/files/' #location of output files

	#dataset short names, types, and colors

	datasets = { 'AcademicRanking' : 'universities', 'AtlasComplex' : 'countries', 'Citations' : 'scientists', 'Cities_RU' : 'cities (RU)', 'Cities_UK' : 'cities (GB)', 'Earthquakes_avgMagnitude' : 'regions JP\n(quake mag)', 'Earthquakes_numberQuakes' : 'regions JP\n(quakes)', 'english' : 'English', 'enron-sent-mails-weekly' : 'Enron emails', 'FIDEFemale' : 'chess players\n(female)', 'FIDEMale' : 'chess players\n(male)', 'Football_FIFA' : 'national football\nteams', 'Football_Scorers' : 'Football scorers', 'Fortune' : 'companies', 'french' : 'French', 'german' : 'German', 'github-watch-weekly' : 'GitHub\nrepositories', 'Golf_OWGR' : 'golf players', 'Hienas' : 'hyenas', 'italian' : 'Italian', 'metroMex' : 'metro stations\n(Mexico)', 'Nascar_BuschGrandNational' : 'Nascar drivers\n(Busch)', 'Nascar_WinstonCupGrandNational' : 'Nascar drivers\n(Winston Cup)', 'Poker_GPI' : 'poker players', 'russian' : 'Russian', 'spanish' : 'Spanish', 'Tennis_ATP' : 'tennis players', 'TheGuardian_avgRecommends' : 'The Guardian\nreaders (recc)', 'TheGuardian_numberComments' : 'The Guardian\nreaders (comm)', 'UndergroundByWeek' : 'metro stations\n(London)' } #name dict

	datatypes = { 'AcademicRanking' : 'open', 'AtlasComplex' : 'open', 'Citations' : 'open', 'Cities_RU' : 'open', 'Cities_UK' : 'closed', 'Earthquakes_avgMagnitude' : 'closed', 'Earthquakes_numberQuakes' : 'closed', 'english' : 'open', 'enron-sent-mails-weekly' : 'open', 'FIDEFemale' : 'open', 'FIDEMale' : 'open', 'Football_FIFA' : 'closed', 'Football_Scorers' : 'open', 'Fortune' : 'open', 'french' : 'open', 'german' : 'open', 'github-watch-weekly' : 'open', 'Golf_OWGR' : 'open', 'Hienas' : 'open', 'italian' : 'open', 'metroMex' : 'closed', 'Nascar_BuschGrandNational' : 'open', 'Nascar_WinstonCupGrandNational' : 'open', 'Poker_GPI' : 'open', 'russian' : 'open', 'spanish' : 'open','Tennis_ATP' : 'open', 'TheGuardian_avgRecommends' : 'open', 'TheGuardian_numberComments' : 'open', 'UndergroundByWeek' : 'closed' } #type dict

	datacats = { 'AcademicRanking' : 'society', 'AtlasComplex' : 'economics', 'Citations' : 'society', 'Cities_RU' : 'infrastructure', 'Cities_UK' : 'infrastructure', 'Earthquakes_avgMagnitude' : 'nature', 'Earthquakes_numberQuakes' : 'nature', 'english' : 'languages', 'enron-sent-mails-weekly' : 'society', 'FIDEFemale' : 'sports', 'FIDEMale' : 'sports', 'Football_FIFA' : 'sports', 'Football_Scorers' : 'sports', 'Fortune' : 'economics', 'french' : 'languages', 'german' : 'languages', 'github-watch-weekly' : 'society', 'Golf_OWGR' : 'sports', 'Hienas' : 'nature', 'italian' : 'languages', 'metroMex' : 'infrastructure', 'Nascar_BuschGrandNational' : 'sports', 'Nascar_WinstonCupGrandNational' : 'sports', 'Poker_GPI' : 'sports', 'russian' : 'languages', 'spanish' : 'languages','Tennis_ATP' : 'sports', 'TheGuardian_avgRecommends' : 'society', 'TheGuardian_numberComments' : 'society', 'UndergroundByWeek' : 'infrastructure' } #type dict

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
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (10, 6),
	'aspect_ratio' : (5, 6),
	'grid_params' : dict( left=0.09, bottom=0.09, right=0.99, top=0.88, wspace=0.3, hspace=0.7 ),
	'dpi' : 300,
	'savename' : 'figure_supp_change' }


	## DATA ##

	#get parameters for all datasets
	params_data = pd.read_pickle( saveloc_data+'params_data.pkl' )

	#prepare dataset order by flux

	fluxmean_data = pd.Series( np.zeros( len(datasets) ), index=datasets.keys(), name='flux_mean' )
	for dataname in datasets: #loop through datasets

		#get parameters for dataset
		params = params_data.loc[ dataname ]
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

		#get flux mean in data
		fluOprops_data = pd.read_pickle( saveloc_data + 'fluOprops_' + param_str_data )
		#average OUT-/IN- flux over time, and then over ranks
		fluxmean_data[ dataname ] = fluOprops_data.mean( axis=0 ).mean()

	fluxmean_data.sort_values( ascending=False, inplace=True ) #sort values


	## PLOTTING ##

	#initialise plot
	sns.set( style="white" ) #set fancy fancy plot
	#convert fig size to inches
	fig = plt.figure( fig_props['fig_num'], figsize=np.array( fig_props['fig_size'] ) )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# A: Rank dependence of rank change in open/closed systems

	leg_lines, leg_labels = [], [] #initialise legend lines/legends lists

	for grid_pos, dataname in enumerate( fluxmean_data.index ): #loop through datasets (in order by decreasing mean flux)
		print( 'flux = {:.2f}, dataset = {}'.format( fluxmean_data[ dataname ], dataname ) ) #to know where we stand

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos ] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [ 24, 25, 26, 27, 28, 29 ]:
			plt.xlabel( '$R / N_0$', size=plot_props['xylabel'], labelpad=2 )
		if grid_pos in [ 0, 6, 12, 18, 24 ]:
			plt.ylabel( r'$C$', size=plot_props['xylabel'], labelpad=2 )

		#prepare data

		#get parameters for dataset
		params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

		#get rank change in data
		rankprops_data = pd.read_pickle( saveloc_data + 'rankprops_' + param_str_data )
		rankchange_data = rankprops_data.loc[ 'rankchange' ]

		#prepare model

		#model fit parameters
		datatype = datatypes[ dataname ] #dataset type: open, closed

		#get model parameters for selected dataset
		params_model = data_misc.data_estimate_params_all( dataname, params, loadflag, saveloc_data, datatype=datatype )
		params[ 'pnu' ], params[ 'ptau' ] = params_model.loc[ 'optimal', [ 'pnu', 'ptau' ] ] #set parameters

		print( 'ptau = {:.4f}, pnu = {:.4f}'.format( params['ptau'], params['pnu'] ) ) #to know where we stand

		rankchange_model = model_misc.change_theo( params )


		#plot plot!

		xplot = np.arange( 1, N0+1 ) / N0 #normalised rank = ( 1, ..., N0 ) / N0

		#unfiltered data
		yplot_data = rankchange_data
		line, = plt.plot( xplot, yplot_data, c=datacols[ datacats[ dataname ] ], lw=plot_props['linewidth'], rasterized=False, zorder=0 )

		#unfiltered model
		yplot_model = rankchange_model
		plt.plot( xplot, yplot_model, '--', c='0.5', lw=plot_props['linewidth']-2, rasterized=False, zorder=1 )

		#texts
		plt.text( 0.5, 1.3, datasets[ dataname ], va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#mean flux
		fluxmean_str = r'$F =$ '+'{0:.2f}'.format( fluxmean_data[ dataname ] )
		plt.text( 0.99, 0.01, fluxmean_str, va='bottom', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'])

		#prepare legend
		if datacats[ dataname ] not in leg_labels:
			leg_lines.append( line )
			leg_labels.append( datacats[ dataname ] )

		#finalise subplot
		plt.axis([ -0.05, 1.05, -0.05, 1.05 ])
		ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )
		if grid_pos not in [ 24, 25, 26, 27, 28, 29 ]:
			plt.xticks([])
		if grid_pos not in [ 0, 6, 12, 18, 24 ]:
			plt.yticks([])
		ax.locator_params( nbins=4 ) #change number of ticks in axes

	#texts and arrows
	plt.text( -7.15, 7.25, 'open', va='center', ha='center', transform=ax.transAxes, weight='bold', rotation='vertical', fontsize=plot_props['xylabel'] )
	plt.text( -7.15, 0.5, 'closed', va='center', ha='center', transform=ax.transAxes, weight='bold', rotation='vertical', fontsize=plot_props['xylabel'] )
	plt.annotate( text='', xy=( -7.15, 1.1 ), xytext=( -7.15, 6.7 ), arrowprops=dict(arrowstyle='<->', color='0.6'), xycoords=ax.transAxes, textcoords=ax.transAxes )

	#legend
	plt.legend( leg_lines, leg_labels, loc='upper left', bbox_to_anchor=(-5.2, 9), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=len(leg_lines) )

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )

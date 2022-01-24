#! /usr/bin/env python

#Farranks - Exploring rank dynamics in complex systems
#Copyright (C) 2022 Gerardo Iñiguez

### SCRIPT FOR PLOTTING SUPP FIGURE (RANK DYNAMICS) IN FARRANKS PROJECT ###

#import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser

## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#subplot settings
	sel_ranks = np.array([ 0.00, 0.25, 0.50, 0.75, 1.00 ]) #selected ranks to show

	#flag and locations
	loadflag = 'y'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	location = root_loc+'diversity/Database/' #location of datasets
	saveloc = root_loc+'nullModel/v4/files/' #location of output files
	figuloc = root_loc+'nullModel/v4/figures/' #location of figure files

	#dataset short names, types, and colors

	datasets = { 'AcademicRanking' : 'universities', 'AtlasComplex' : 'countries', 'Citations' : 'scientists', 'Cities_RU' : 'cities (RU)', 'Cities_UK' : 'cities (GB)', 'Earthquakes_avgMagnitude' : 'regions JP\n(quake mag)', 'Earthquakes_numberQuakes' : 'regions JP\n(quakes)', 'english' : 'English', 'enron-sent-mails-weekly' : 'Enron emails', 'FIDEFemale' : 'chess players\n(female)', 'FIDEMale' : 'chess players\n(male)', 'Football_FIFA' : 'national football\nteams', 'Football_Scorers' : 'Football scorers', 'Fortune' : 'companies', 'french' : 'French', 'german' : 'German', 'github-watch-weekly' : 'GitHub\nrepositories', 'Golf_OWGR' : 'golf players', 'Hienas' : 'hyenas', 'italian' : 'Italian', 'metroMex' : 'metro stations\n(Mexico)', 'Nascar_BuschGrandNational' : 'Nascar drivers\n(Busch)', 'Nascar_WinstonCupGrandNational' : 'Nascar drivers\n(Winston Cup)', 'Poker_GPI' : 'poker players', 'russian' : 'Russian', 'spanish' : 'Spanish', 'Tennis_ATP' : 'tennis players', 'TheGuardian_avgRecommends' : 'The Guardian\nreaders (recc)', 'TheGuardian_numberComments' : 'The Guardian\nreaders (comm)', 'UndergroundByWeek' : 'metro stations\n(London)' } #name dict

	datatypes = { 'AcademicRanking' : 'society', 'AtlasComplex' : 'economics', 'Citations' : 'society', 'Cities_RU' : 'infrastructure', 'Cities_UK' : 'infrastructure', 'Earthquakes_avgMagnitude' : 'nature', 'Earthquakes_numberQuakes' : 'nature', 'english' : 'languages', 'enron-sent-mails-weekly' : 'society', 'FIDEFemale' : 'sports', 'FIDEMale' : 'sports', 'Football_FIFA' : 'sports', 'Football_Scorers' : 'sports', 'Fortune' : 'economics', 'french' : 'languages', 'german' : 'languages', 'github-watch-weekly' : 'society', 'Golf_OWGR' : 'sports', 'Hienas' : 'nature', 'italian' : 'languages', 'metroMex' : 'infrastructure', 'Nascar_BuschGrandNational' : 'sports', 'Nascar_WinstonCupGrandNational' : 'sports', 'Poker_GPI' : 'sports', 'russian' : 'languages', 'spanish' : 'languages','Tennis_ATP' : 'sports', 'TheGuardian_avgRecommends' : 'society', 'TheGuardian_numberComments' : 'society', 'UndergroundByWeek' : 'infrastructure' } #type dict


	palette = sns.color_palette( 'Pastel1', n_colors=7 ) #selected colormap for types
	datacols = { 'society' : palette[0], 'economics' : palette[1], 'nature' : palette[2], 'infrastructure' : palette[3], 'sports' : palette[4], 'languages' : palette[6] } #set color for dataset type

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 13,
	'text_size' : 10,
	'marker_size' : 3,
	'linewidth' : 2,
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
	'grid_params' : dict( left=0.09, bottom=0.07, right=0.99, top=0.845, wspace=0.2, hspace=0.8 ),
	'dpi' : 300,
	'savename' : 'figure_supp_dynamics' }

	#get colors
	colors = sns.color_palette( 'coolwarm', n_colors=len( sel_ranks ) )


	## DATA ##

	#get parameters for all datasets
	params_data = pd.read_pickle( saveloc+'params_data.pkl' )

	#prepare dataset order by flux

	fluxmean_data = pd.Series( np.zeros( len(datasets) ), index=datasets.keys(), name='flux_mean' )
	for dataname in datasets: #loop through datasets

		#get parameters for dataset
		params = params_data.loc[ dataname ]
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		param_str = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

		#get flux mean in data
		fluOprops_data = pd.read_pickle( saveloc + 'fluOprops_' + param_str )
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


# A: Rank evolution of elements in open/closed systems

	for grid_pos, dataname in enumerate( fluxmean_data.index ): #loop through datasets (in order by decreasing mean flux)
		print( 'flux = {:.2f}, dataset = {}'.format( fluxmean_data[ dataname ], dataname ) ) #to know where we stand

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos ] )
		sns.despine( ax=ax, top=False, bottom=True ) #take out bottom and right spines
		if grid_pos in [ 0, 1, 2, 3, 4, 5 ]:
			plt.xlabel( '$t / T$', size=plot_props['xylabel'], labelpad=4 )
		if grid_pos in [ 0, 6, 12, 18, 24 ]:
			plt.ylabel( '$R_t / N_0$', size=plot_props['xylabel'], labelpad=2 )
		ax.xaxis.set_label_position( 'top' ) #set x-axis label on top
		ax.invert_yaxis() #invert y-axis

		#prepare data

		#get parameters for dataset
		params = params_data.loc[ dataname ]
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		param_str = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

		#get rank/element time series in data
		savenames = ( saveloc + 'rankseries_' + param_str, saveloc + 'elemseries_' + param_str )
		rankseries = pd.read_pickle( savenames[0] )
		elemseries = pd.read_pickle( savenames[1] )

		#plot plot!

		xplot = rankseries.index / float( T ) #normalised time = ( 0, ..., T-1 ) / T

		for pos, rank in enumerate( ( (N0-1) * sel_ranks ).astype(int) ): #loop through selected ranks
			elem = elemseries.at[ T-1, rank ] #find element in rank at t=T-1
			print( elem ) #to know where we stand

			#get (normalised) rank time series of element
			yplot = rankseries[ elem ] + 1 #original (displaced) ranks = 1, ..., N0
			yplot = yplot / N0 #and normalise

			label = str( sel_ranks[ pos ] ) #labels for legend
			if pos == 0:
				label = '$R_{T - 1} / N_0 = $ ' + label #fix 1st label

			plt.plot( xplot, yplot, label=label, c=colors[ pos ], lw=plot_props['linewidth'], rasterized=False )

		#texts and lines

		if grid_pos < 6:
			plt.text( 0.5, 2.2, datasets[ dataname ], va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )
		else:
			plt.text( 0.5, 1.35, datasets[ dataname ], va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		if grid_pos < 24:
			plt.axhline( y=1, c='0.7', ls='--', lw=plot_props['linewidth'] )

		#mean flux
		fluxmean_str = r'$F =$ '+'{0:.2f}'.format( fluxmean_data[ dataname ] )
		plt.text( 0.99, 0.1, fluxmean_str, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'])

		#finalise subplot
		plt.axis([ -0.05, 1.05, 1.2, -0.05 ])
		ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=-2 )
		if grid_pos not in [ 0, 1, 2, 3, 4, 5 ]:
			plt.xticks([])
		if grid_pos not in [ 0, 6, 12, 18, 24 ]:
			plt.yticks([])
		ax.locator_params( nbins=4 ) #change number of ticks in axes

	#texts and arrows
	plt.text( -6.6, 7.6, 'open', va='center', ha='center', transform=ax.transAxes, weight='bold', rotation='vertical', fontsize=plot_props['xylabel'] )
	plt.text( -6.6, 0.5, 'closed', va='center', ha='center', transform=ax.transAxes, weight='bold', rotation='vertical', fontsize=plot_props['xylabel'] )
	plt.annotate( s='', xy=( -6.6, 1.1 ), xytext=( -6.6, 7.1 ), arrowprops=dict(arrowstyle='<->', color='0.6'), xycoords=ax.transAxes, textcoords=ax.transAxes )

	#legend
	leg = plt.legend( loc='lower center', bbox_to_anchor=( -2.5, -0.8 ), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], ncol=len( sel_ranks ), columnspacing=plot_props[ 'legend_colsp' ] )

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )

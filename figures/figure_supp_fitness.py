#! /usr/bin/env python

### SCRIPT FOR PLOTTING SUPP FIGURE (FITNESS) IN FARRANKS PROJECT ###

#import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser
from matplotlib.ticker import ( MultipleLocator, LogLocator )


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#subplot settings
	sel_times = np.array([ 0.00, 0.25, 0.50, 0.75, 1.00 ]) #selected times to show

	#flag and locations
	loadflag = 'y'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc = root_loc+'nullModel/v4/files/' #location of output files
	figuloc = root_loc+'nullModel/v4/figures/' #location of figure files

	#dataset short names, types, and colors

	datasets = { 'AcademicRanking' : 'universities', 'AtlasComplex' : 'countries', 'Citations' : 'scientists', 'Cities_RU' : 'cities (RU)', 'Cities_UK' : 'cities (GB)', 'Earthquakes_avgMagnitude' : 'regions JP\n(quake mag)', 'Earthquakes_numberQuakes' : 'regions JP\n(quakes)', 'english' : 'English', 'enron-sent-mails-weekly' : 'Enron emails', 'FIDEFemale' : 'chess players\n(female)', 'FIDEMale' : 'chess players\n(male)', 'Football_FIFA' : 'national football\nteams', 'Football_Scorers' : 'Football scorers', 'Fortune' : 'companies', 'french' : 'French', 'german' : 'German', 'github-watch-weekly' : 'GitHub\nrepositories', 'Golf_OWGR' : 'golf players', 'Hienas' : 'hyenas', 'italian' : 'Italian', 'metroMex' : 'metro stations\n(Mexico)', 'Nascar_BuschGrandNational' : 'Nascar drivers\n(Busch)', 'Nascar_WinstonCupGrandNational' : 'Nascar drivers\n(Winston Cup)', 'Poker_GPI' : 'poker players', 'russian' : 'Russian', 'spanish' : 'Spanish', 'Tennis_ATP' : 'tennis players', 'TheGuardian_avgRecommends' : 'The Guardian\nreaders (recc)', 'TheGuardian_numberComments' : 'The Guardian\nreaders (comm)', 'UndergroundByWeek' : 'metro stations\n(London)' } #name dict
#	datasets = { 'VideogameEarnings' : 'videogame\nplayers', 'Virus' : 'viruses' } #shady data

	datatypes = { 'AcademicRanking' : 'society', 'AtlasComplex' : 'economics', 'Citations' : 'society', 'Cities_RU' : 'infrastructure', 'Cities_UK' : 'infrastructure', 'Earthquakes_avgMagnitude' : 'nature', 'Earthquakes_numberQuakes' : 'nature', 'english' : 'languages', 'enron-sent-mails-weekly' : 'society', 'FIDEFemale' : 'sports', 'FIDEMale' : 'sports', 'Football_FIFA' : 'sports', 'Football_Scorers' : 'sports', 'Fortune' : 'economics', 'french' : 'languages', 'german' : 'languages', 'github-watch-weekly' : 'society', 'Golf_OWGR' : 'sports', 'Hienas' : 'nature', 'italian' : 'languages', 'metroMex' : 'infrastructure', 'Nascar_BuschGrandNational' : 'sports', 'Nascar_WinstonCupGrandNational' : 'sports', 'Poker_GPI' : 'sports', 'russian' : 'languages', 'spanish' : 'languages','Tennis_ATP' : 'sports', 'TheGuardian_avgRecommends' : 'society', 'TheGuardian_numberComments' : 'society', 'UndergroundByWeek' : 'infrastructure' } #type dict
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
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (10, 6),
	'aspect_ratio' : (5, 6),
	'grid_params' : dict( left=0.095, bottom=0.09, right=0.99, top=0.88, wspace=0.3, hspace=0.7 ),
	'dpi' : 300,
	'savename' : 'figure_supp_fitness' }

	#get colors
	colors = sns.color_palette( 'GnBu', n_colors=len( sel_times ) )


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


# A: Fitness (score) distribution in open/closed systems

	for grid_pos, dataname in enumerate( fluxmean_data.index ): #loop through datasets (in order by decreasing mean flux)
		print( 'flux = {:.2f}, dataset = {}'.format( fluxmean_data[ dataname ], dataname ) ) #to know where we stand

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos ] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [ 24, 25, 26, 27, 28, 29 ]:
			plt.xlabel( '$f$', size=plot_props['xylabel'], labelpad=2 )
		if grid_pos in [ 0, 6, 12, 18, 24 ]:
			plt.ylabel( r'$P_f / P_m$', size=plot_props['xylabel'], labelpad=2 )

		#prepare data

		#get parameters for dataset
		params = params_data.loc[ dataname ]
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		param_str = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

		#get score time series in data
		scorseries = pd.read_pickle( saveloc + 'scorseries_' + param_str )

		#plot plot!

		for pos_t, t in enumerate( ( (T-1) * sel_times ).astype(int) ): #loop through selected times

			#get fitness as score (normalised by maximum score)
			fitness = scorseries.loc[ t ] / scorseries.loc[ t ].max()

			#get log bins from min/max fitness
			bin_min = np.log10( 0.99 * ( fitness[ fitness > 0 ].min() ) )
			bin_max = np.log10( 1.01 * fitness.max() )
			bin_edges = np.logspace( bin_min, bin_max, 10 )

			#calculate histogram of scores (for given time t)
			yplot, bin_edges = np.histogram( fitness, bins=bin_edges, density=True )
			yplot = pd.DataFrame(yplot)
			yplot = yplot.mask( yplot <= 0 ) #mask bins with no counts
			yplot = yplot / yplot.max() #normalise by maximum value
			yplot = yplot.to_numpy()

			xplot = [] #initialise array of bin centers
			for pos_bin, bin in enumerate( bin_edges[:-1] ): #and get them
				xplot.append( ( bin + bin_edges[ pos_bin + 1 ] ) / 2 )

			label = str( sel_times[ pos_t ] ) #labels for legend
			if pos_t == 0:
				label = '$t / (T - 1) = $ ' + label #fix 1st label

			#plot plot!
			plt.loglog( xplot, yplot, '-', label=label, c=colors[ pos_t ], lw=plot_props['linewidth'], rasterized=False )

		#texts
		plt.text( 0.5, 1.3, datasets[ dataname ], va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#mean flux
		fluxmean_str = r'$F =$ '+'{0:.2f}'.format( fluxmean_data[ dataname ] )
		plt.text( 0.01, 0.01, fluxmean_str, va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['text_size'])

		#finalise subplot
		plt.axis([ 0.0001, 1.5, 0.000001, 5 ])
		ax.xaxis.set_major_locator( LogLocator( numticks=3 ) )
		ax.yaxis.set_major_locator( LogLocator( numticks=3 ) )
		ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )
		if grid_pos not in [ 24, 25, 26, 27, 28, 29 ]:
			plt.xticks([])
		if grid_pos not in [ 0, 6, 12, 18, 24 ]:
			plt.yticks([])

	#texts and arrows
	plt.text( -7.19, 7.25, 'open', va='center', ha='center', transform=ax.transAxes, weight='bold', rotation='vertical', fontsize=plot_props['xylabel'] )
	plt.text( -7.19, 0.5, 'closed', va='center', ha='center', transform=ax.transAxes, weight='bold', rotation='vertical', fontsize=plot_props['xylabel'] )
	plt.annotate( text='', xy=( -7.19, 1.1 ), xytext=( -7.19, 6.7 ), arrowprops=dict(arrowstyle='<->', color='0.6'), xycoords=ax.transAxes, textcoords=ax.transAxes )

	#legend
	plt.legend( loc='upper left', bbox_to_anchor=(-4.4, 9), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=len( sel_times ) )

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )

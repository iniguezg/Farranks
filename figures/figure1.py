#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE 1 IN FARRANKS PROJECT ###

#import modules
import os
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.signal as ss
import svgutils.compose as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#flag and locations
	loadflag = 'y'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	location = root_loc+'diversity/Database/' #location of datasets
	saveloc_data = root_loc+'nullModel/v4/files/' #location of output files
	figuloc = root_loc+'nullModel/v4/figures/' #location of figure files

	#dataset short names, types, and colors

	datasets = { 'AcademicRanking' : 'universities', 'AtlasComplex' : 'countries', 'Citations' : 'scientists', 'Cities_RU' : 'cities (RU)', 'Cities_UK' : 'cities (GB)', 'Earthquakes_avgMagnitude' : 'regions JP\n(quake mag)', 'Earthquakes_numberQuakes' : 'regions JP\n(quakes)', 'english' : 'English', 'enron-sent-mails-weekly' : 'Enron emails', 'FIDEFemale' : 'chess players\n(female)', 'FIDEMale' : 'chess players\n(male)', 'Football_FIFA' : 'national football\nteams', 'Football_Scorers' : 'Football scorers', 'Fortune' : 'companies', 'french' : 'French', 'german' : 'German', 'github-watch-weekly' : 'GitHub\nrepositories', 'Golf_OWGR' : 'golf players', 'Hienas' : 'hyenas', 'italian' : 'Italian', 'metroMex' : 'metro stations\n(Mexico)', 'Nascar_BuschGrandNational' : 'Nascar drivers\n(Busch)', 'Nascar_WinstonCupGrandNational' : 'Nascar drivers\n(Winston Cup)', 'Poker_GPI' : 'poker players', 'russian' : 'Russian', 'spanish' : 'Spanish', 'Tennis_ATP' : 'tennis players', 'TheGuardian_avgRecommends' : 'The Guardian\nreaders (recc)', 'TheGuardian_numberComments' : 'The Guardian\nreaders (comm)', 'UndergroundByWeek' : 'metro stations\n(London)' } #name dict
#	datasets = { 'VideogameEarnings' : 'videogame\nplayers', 'Virus' : 'viruses' } #shady data

	datasets_oneliners = { 'AcademicRanking' : 'universities', 'AtlasComplex' : 'countries', 'Citations' : 'scientists', 'Cities_RU' : 'cities (RU)', 'Cities_UK' : 'cities (GB)', 'Earthquakes_avgMagnitude' : 'regions JP (quake mag)', 'Earthquakes_numberQuakes' : 'regions JP(quakes)', 'english' : 'English', 'enron-sent-mails-weekly' : 'Enron emails', 'FIDEFemale' : 'chess players (female)', 'FIDEMale' : 'chess players (male)', 'Football_FIFA' : 'national football teams', 'Football_Scorers' : 'Football scorers', 'Fortune' : 'companies', 'french' : 'French', 'german' : 'German', 'github-watch-weekly' : 'GitHub repositories', 'Golf_OWGR' : 'golf players', 'Hienas' : 'hyenas', 'italian' : 'Italian', 'metroMex' : 'metro stations (Mexico)', 'Nascar_BuschGrandNational' : 'Nascar drivers (Busch)', 'Nascar_WinstonCupGrandNational' : 'Nascar drivers (Winston Cup)', 'Poker_GPI' : 'poker players', 'russian' : 'Russian', 'spanish' : 'Spanish', 'Tennis_ATP' : 'tennis players', 'TheGuardian_avgRecommends' : 'The Guardian readers (recc)', 'TheGuardian_numberComments' : 'The Guardian readers (comm)', 'UndergroundByWeek' : 'metro stations (London)' } #name dict
#	datasets_oneliners = { 'VideogameEarnings' : 'videogame players', 'Virus' : 'viruses' } #shady data

	datasets_openclosed = { 'open' : [ 'AcademicRanking', 'AtlasComplex', 'Citations', 'Cities_RU', 'english', 'enron-sent-mails-weekly', 'FIDEFemale', 'FIDEMale', 'Football_Scorers', 'Fortune', 'french', 'german', 'github-watch-weekly', 'Hienas', 'italian', 'Nascar_BuschGrandNational', 'Nascar_WinstonCupGrandNational', 'Poker_GPI', 'russian', 'spanish', 'Tennis_ATP', 'TheGuardian_avgRecommends', 'TheGuardian_numberComments' ],
							'closed' : [ 'Cities_UK', 'Earthquakes_avgMagnitude',
	'Earthquakes_numberQuakes', 'Football_FIFA', 'Golf_OWGR', 'metroMex', 'UndergroundByWeek' ] }

	datatypes = { 'AcademicRanking' : 'society', 'AtlasComplex' : 'economics', 'Citations' : 'society', 'Cities_RU' : 'infrastructure', 'Cities_UK' : 'infrastructure', 'Earthquakes_avgMagnitude' : 'nature', 'Earthquakes_numberQuakes' : 'nature', 'english' : 'languages', 'enron-sent-mails-weekly' : 'society', 'FIDEFemale' : 'sports', 'FIDEMale' : 'sports', 'Football_FIFA' : 'sports', 'Football_Scorers' : 'sports', 'Fortune' : 'economics', 'french' : 'languages', 'german' : 'languages', 'github-watch-weekly' : 'society', 'Golf_OWGR' : 'sports', 'Hienas' : 'nature', 'italian' : 'languages', 'metroMex' : 'infrastructure', 'Nascar_BuschGrandNational' : 'sports', 'Nascar_WinstonCupGrandNational' : 'sports', 'Poker_GPI' : 'sports', 'russian' : 'languages', 'spanish' : 'languages','Tennis_ATP' : 'sports', 'TheGuardian_avgRecommends' : 'society', 'TheGuardian_numberComments' : 'society', 'UndergroundByWeek' : 'infrastructure' } #type dict
#	datasets = { 'VideogameEarnings' : 'economics', 'Virus' : 'nature' } #shady data

	palette = sns.color_palette( 'Set2', n_colors=7 ) #selected colormap for types
	datacols = { 'society' : palette[0], 'languages' : palette[1], 'economics' : palette[2], 'infrastructure' : palette[3], 'nature' : palette[4], 'sports' : palette[6] } #set color for dataset type

	systems = {} #initialise systems dict
	for type in set( datatypes.values() ): #loop through types
		systems[ type ] = [] #initialise list of datasets per type
	for dataname, type in datatypes.items(): #loop through datasets
		systems[ type ].append( dataname ) #classify datasets by type

	#sizes/widths/coords
	plot_props = { 'xylabel' : 13,
	'figlabel' : 26,
	'ticklabel' : 13,
	'text_size' : 15,
	'marker_size' : 8,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':11 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables

	fig_props = { 'fig_num' : 1,
	'fig_size' : [ 30.48, 15.24 ],
	'aspect_ratio' : (3, 3),
	'grid_params' : dict( left=0.045, bottom=0.075, right=0.985, top=0.965, wspace=0.4, hspace=0.6 ),
	'height_ratios' : [1, 0.7, 0.7],
	'savename' : 'fig1_bot' } #for bottom figure

	svg_props = { 'fig_size' : [ '30.48cm', '15.24cm' ],
	'dpi' : '300',
	'added_file' : 'drawings/fig1_top',
	'savename' : 'figure1' } #for whole figure


	## DATA ##

	#get parameters for all datasets
	params_data = pd.read_pickle( saveloc_data+'params_data.pkl' )


	## PLOTTING ##

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	#convert fig size to inches
	fig = plt.figure( fig_props['fig_num'], figsize=np.array( fig_props['fig_size'] ) / 2.54 )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'], height_ratios=fig_props['height_ratios'] )
	grid.update( **fig_props['grid_params'] )


# B1: Time evolution of rank openness in open/closed systems

	#initialise subplot
	subgrid = grid[ 0, 1 ].subgridspec( 2, 1, hspace=0, height_ratios=[0.15, 1] )
	ax = plt.subplot( subgrid[ 1 ] )
	sns.despine( ax=ax ) #take out top and right spines
	plt.xlabel( '$t$', size=plot_props['xylabel'], labelpad=0 )
	plt.ylabel( '$o_t$', size=plot_props['xylabel'], labelpad=3 )

	for type in [ 'society', 'languages', 'economics', 'infrastructure', 'nature', 'sports' ]: #loop through types (ordered like Table S1)
		for pos, dataname in enumerate( systems[ type ] ): #and their datasets

			#get parameters for dataset
			params = params_data.loc[ dataname ]
			N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
			param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

			#load openness in data
			openprops_data = pd.read_pickle( saveloc_data + 'openprops_' + param_str_data )
			openness_data = openprops_data.loc[ 'openness' ]

			#plot plot!
			label = type if pos == 0 else '_nolegend_'
			plt.plot( openness_data.index, openness_data, label=label, c=datacols[ type ], lw=plot_props['linewidth'] )

	#texts
	plt.text( 0.65, 0.8, 'more open', va='center', ha='center', transform=ax.transAxes, weight='bold', fontsize=plot_props['ticklabel'] )
	plt.text( 0.65, 0.09, 'less open', va='center', ha='center', transform=ax.transAxes, weight='bold', fontsize=plot_props['ticklabel'] )
	plt.annotate( text='', xy=( 0.65, 0.75 ), xytext=( 0.65, 0.25 ), arrowprops=dict(arrowstyle='<->', color='0.6'), xycoords=ax.transAxes, textcoords=ax.transAxes )

	#legend
	leg = plt.legend( loc='lower left', bbox_to_anchor=(0.1, 1.04), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=len(systems) )

	#finalise subplot
	plt.axis([ 0, 50, 0, 5 ])
	ax.tick_params( axis='both', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=2 )
	ax.locator_params( axis='x', nbins=5 )
	ax.locator_params( axis='y', nbins=5 )


# B2: Mean flux vs. avg openness derivatice in open/closed systems

	sel_datasets = [ 'AcademicRanking', 'english', 'Cities_UK', 'metroMex' ] #open/closed examples

	#label formats: x offset, y offset, va, ha
	formats = {
	'AcademicRanking' : ( -20, 10, 'bottom', 'right' ),
	'english' : ( 7, -15, 'top', 'center' ),
	'Cities_UK' : ( 32, 24, 'bottom', 'center' ),
	'metroMex' : ( 28, 2, 'center', 'left' ) }

	#subplot variables
	xlims = [ -2e-4, 2e-3 ] #limits for linear region in axes
	ylims = [ -6e-5, 2e-4 ]

	#initialise subplot
	subgrid = grid[ 0, 2 ].subgridspec( 2, 1, hspace=0, height_ratios=[ 0.15, 1 ] )
	ax = plt.subplot( subgrid[ 1 ] )
	sns.despine( ax=ax ) #take out top and right spines
	plt.xlabel( '$F$', size=plot_props['xylabel'], labelpad=0 )
	plt.ylabel( r'$\dot{o}$', size=plot_props['xylabel'], labelpad=0 )

	for type in [ 'society', 'languages', 'economics', 'infrastructure', 'nature', 'sports' ]: #loop through types (ordered like Table S1)
		for pos, dataname in enumerate( systems[ type ] ): #and their datasets

			#get parameters for dataset
			params = params_data.loc[ dataname ]
			N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
			param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

			#mean rank flux
			fluOprops_data = pd.read_pickle( saveloc_data + 'fluOprops_' + param_str_data )
			fluxmean_data = fluOprops_data.mean( axis=0 ).mean() #average OUT-/IN- flux over time, and then over ranks

			#avg openness derivative
			openprops_data = pd.read_pickle( saveloc_data + 'openprops_' + param_str_data )
			openmean_data = openprops_data.loc[ 'open_deriv' ].mean() #average derivative over time

			#plot plot!
			plt.plot( fluxmean_data, openmean_data, 'o', label=None, c=datacols[ type ], ms=plot_props['marker_size'] )

			#names/arrows for selected datasets
			if dataname in sel_datasets:
				plt.annotate( text=datasets_oneliners[ dataname ], xy=( fluxmean_data, openmean_data ), xytext=( formats[dataname][0], formats[dataname][1]), va=formats[dataname][2], ha=formats[dataname][3], arrowprops=dict( headlength=1, headwidth=1, width=0.5, color=datacols[ type ] ), textcoords='offset points', size=12, color='0.4', zorder=1 )

	#lines
	plt.hlines( ylims[1], xlims[0], xlims[1], ls='--', colors='0.6', lw=1.5, zorder=0 )
	plt.vlines( xlims[1], ylims[0], ylims[1], ls='--', colors='0.6', lw=1.5, zorder=0 )

	#texts
	# plt.text( 0.8, 1, 'open systems', va='center', ha='center', transform=ax.transAxes, weight='bold', fontsize=plot_props['ticklabel'] )
	# plt.text( 0.25, 0.07, 'closed systems', va='center', ha='center', transform=ax.transAxes, weight='bold', fontsize=plot_props['ticklabel'] )

	#finalise subplot
	plt.axis([ xlims[0], 1e0, ylims[0], 1e0 ])
	plt.xscale( 'symlog', linthresh=xlims[1] )
	plt.yscale( 'symlog', linthresh=ylims[1] )
	ax.tick_params( axis='x', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.tick_params( axis='y', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=2 )
	plt.minorticks_off()


# C: Rank evolution of elements in open/closed systems

	sel_datasets = [ 'AcademicRanking', 'english', 'Cities_UK', 'metroMex' ] #open/closed examples
	sel_tuples = [ (0, 1), (0, 3), (1, 1), (1, 3) ] #plot positions
	sel_labels = { 'AcademicRanking' : [ 'Harvard', 'Duke', "King's College", 'UC Davis', 'Utah' ],
	 			   'metroMex' : [ 'Indios Verdes', 'Múzquiz', 'Iztacalco', 'Refinería', 'Dep. 18 Marzo' ] } 	#selected short names for ranks
	sel_ranks = np.array([ 0.00, 0.25, 0.50, 0.75, 1.00 ]) #selected ranks to show

	#get colors
	colors = sns.color_palette( 'coolwarm', n_colors=len( sel_ranks ) )

	subgrid = grid[ 1:, :2 ].subgridspec( 2, 5, hspace=0.55, wspace=0, width_ratios=[0.1, 1, 0.8, 1, 0.5] )

	for grid_pos, grid_tuple in enumerate( sel_tuples ): #loop through subplot

		#initialise subplot
		ax = plt.subplot( subgrid[ grid_tuple ] )
		sns.despine( ax=ax )
		if grid_pos == 3:
			plt.xlabel( '$t$ (year 2014)', size=plot_props['xylabel'], labelpad=1 )
		else:
			plt.xlabel( '$t$ (year)', size=plot_props['xylabel'], labelpad=1 )
		if grid_pos in [0, 2]:
			plt.ylabel( '$R_t / N_0$', size=plot_props['xylabel'], labelpad=0 )
		ax.invert_yaxis() #invert y-axis

		#prepare data

		#select dataset
		dataname = sel_datasets[ grid_pos ]
		print( '\t'+dataname ) #to know where we stand

		#get parameters for dataset
		params = params_data.loc[ dataname ]
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

		#set limits for plots (with T if needed)
		sel_limits = { 'AcademicRanking' : [ 2003, 2016, 1.05, -0.05 ],
				   'english' : [ 1899, 2009, 1.05, -0.05 ],
				   'Cities_UK' : [ 1900, 2002, 1.05, -0.05 ],
				   'metroMex' : [ 0, T-1, 1.05, -0.05 ] }

		#load list of times
		timelist = pd.read_csv( location+dataname+'/time_list', header=None, names=['time'] )

		#get rank/element time series in data
		rankseries = pd.read_pickle( saveloc_data + 'rankseries_' + param_str_data )
		elemseries = pd.read_pickle( saveloc_data + 'elemseries_' + param_str_data )

		#plot plot!

		if grid_pos == 3: #for not-year-only times
			xplot = rankseries.index #time = ( 0, ..., T-1 )
		else:
			xplot = timelist

		for pos, rank in enumerate( ( (N0-1) * sel_ranks ).astype(int) ): #loop through selected ranks
			elem = elemseries.at[ T-1, rank ] #find element in rank at t=T-1
			print( elem ) #to know where we stand

			#get (normalised) rank time series of element
			yplot = rankseries[ elem ] #original time series
			yplot = yplot.mask( yplot >= N0 ) #mask elements outside ranking (N0, ..., N-1)
			yplot = yplot / float( N0 ) #and normalise

			#labels for legend
			if grid_pos in [ 0, 3 ]: #for badly formatted names
				label = sel_labels[ dataname ][ pos ]
			else:
				label = elem

			plt.plot( xplot, yplot, label=label, c=colors[ pos ], lw=plot_props['linewidth'], zorder=len(sel_ranks)-pos )

		#lines and texts

		plt.text( 1, 1, datasets_oneliners[ dataname ], va='bottom', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'], zorder=len(sel_ranks)+1 )

		if grid_pos == 0:
			plt.text( -0.3, 0.5, 'more open', va='center', ha='center', transform=ax.transAxes, rotation='vertical', weight='bold', fontsize=plot_props['ticklabel'] )
		if grid_pos == 2:
			plt.text( -0.3, 0.5, 'less open', va='center', ha='center', transform=ax.transAxes, rotation='vertical', weight='bold', fontsize=plot_props['ticklabel'] )
			plt.annotate( text='', xy=( -0.3, 1.6 ), xytext=( -0.3, 0.9 ), arrowprops=dict(arrowstyle='<->', color='0.6'), xycoords=ax.transAxes, textcoords=ax.transAxes )

		#legend
		leg = plt.legend( loc='upper left', bbox_to_anchor=(0.98, 1.02), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], ncol=1, columnspacing=plot_props[ 'legend_colsp' ] )

		#finalise subplot
		plt.axis( sel_limits[ dataname ] )
		ax.tick_params( axis='both', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=2 )
		ax.locator_params( nbins=4 ) #change number of ticks in axes
		if grid_pos in [1, 3]:
			plt.yticks([])

		#tick labels (as real dates)
		if grid_pos == 3:
			xlocs, not_used = plt.xticks() #get initial ticks
			xlabels = [] #initialise new labels
			for xloc in xlocs[ :-1 ]: #loop through initial ticks (except last one)
				xlabel = str( timelist.iat[ int(xloc), 0 ] ) #set new label from timelist
				xlabels.append( xlabel[ 3: ] ) #NOTE: only use month (last letters)
			plt.xticks( xlocs[ :-1 ], xlabels ) #set new tick labels


# D : Shape of rank change in open/closed systems

	leg_lines, leg_labels = [], [] #initialise legend elements

	for openclosed_flag in [ 'open', 'closed' ]: #loop through open/closed system groups

		#initialise subplot
		if openclosed_flag == 'open':
			subgrid = grid[ 1, 2 ].subgridspec( 1, 2, wspace=0, width_ratios=[1, 0.08] )
		else:
			subgrid = grid[ 2, 2 ].subgridspec( 1, 2, wspace=0, width_ratios=[1, 0.08] )
		ax = plt.subplot( subgrid[ 0 ] )
		sns.despine( ax=ax ) #take out bottom and right spines
		if openclosed_flag == 'closed':
			plt.xlabel( '$R / N_0$', size=plot_props['xylabel'], labelpad=2 )
		plt.ylabel( '$C$', size=plot_props['xylabel'], labelpad=2 )

		for type in [ 'society', 'languages', 'economics', 'infrastructure', 'nature', 'sports' ]: #loop through types (ordered like Table S1)
			for pos, dataname in enumerate( systems[ type ] ): #and their datasets
				if dataname in datasets_openclosed[ openclosed_flag ]: #only appropriate datasets!

					#get parameters for dataset
					params = params_data.loc[ dataname ]
					N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
					param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

					#load rank change in data
					rankprops_data = pd.read_pickle( saveloc_data + 'rankprops_' + param_str_data )
					rankchange_data = rankprops_data.loc[ 'rankchange' ]

					#plot plot!

					xplot = np.arange( 1, N0+1 ) / N0 #normalised rank = ( 1, ..., N0 ) / N0
					filter_win = int( np.ceil( (N0 / 50 ) / 2 ) * 2 ) + 1 #filter window len (odd int)
					yplot = ss.savgol_filter( rankchange_data, filter_win, 2 )

					line, = plt.plot( xplot, yplot, c=datacols[ type ], lw=plot_props['linewidth'] )

					#legend elements
					if ( openclosed_flag == 'open' or type == 'nature' ) and pos == 0:
						leg_lines.append( line )
						leg_labels.append( type )

		#texts
		if openclosed_flag == 'open':
			plt.text( 1.08, 0.5, 'more open', va='center', ha='center', transform=ax.transAxes, rotation='vertical', weight='bold', fontsize=plot_props['ticklabel'] )
		else:
			plt.text( 1.08, 0.5, 'less open', va='center', ha='center', transform=ax.transAxes, rotation='vertical', weight='bold', fontsize=plot_props['ticklabel'] )
			plt.annotate( text='', xy=( 1.08, 1.7 ), xytext=( 1.08, 0.95 ), arrowprops=dict(arrowstyle='<->', color='0.6'), xycoords=ax.transAxes, textcoords=ax.transAxes )

		#finalise subplot
		plt.axis([ -0.02, 1.02, -0.02, 1.02 ])
		ax.tick_params( axis='both', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=2 )
		if openclosed_flag == 'open':
			plt.xticks([])
		ax.locator_params( axis='x', nbins=5 ) #change number of ticks in axes

	#legend
	leg = plt.legend( leg_lines, leg_labels, loc='lower left', bbox_to_anchor=(-0.2, 1.07), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=3 )

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.svg', format='svg' )
#	if fig_props['savename'] != '':
#		plt.savefig( fig_props['savename']+'.pdf', format='pdf' )


	## SVG ##

	if svg_props['savename'] != '':

		#use svgutils to compose final figure
		sc.Figure( *svg_props['fig_size'],

		sc.Panel( sc.SVG( fig_props['savename']+'.svg' ),
			  sc.Text( 'b', 310, 18, size=plot_props['figlabel']-5, weight='bold' ),
			  sc.Text( 'c', 3, 175, size=plot_props['figlabel']-5, weight='bold' ),
			  sc.Text( 'd', 590, 175, size=plot_props['figlabel']-5, weight='bold' )
		).scale( 1.335 ),

		sc.Panel( sc.SVG( svg_props['added_file']+'.svg' ),
			  sc.Text( 'a', 3, 18, size=plot_props['figlabel'], weight='bold' )
		).scale( 1.05 ),

#		sc.Grid( 20, 20 )

		).save( svg_props['savename']+'.svg' )

		#save final pdf figure
		os.system( 'inkscape --export-dpi='+svg_props['dpi']+' --export-filename='+figuloc+svg_props['savename']+'.pdf '+figuloc+svg_props['savename']+'.svg' )

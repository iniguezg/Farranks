#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE 2 IN FARRANKS PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.signal as ss
import svgutils.compose as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc, model_misc


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#flag and locations
	loadflag = 'y'
	saveflag = 'y'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_data = root_loc+'nullModel/v4/files/' #output files
	saveloc_model = root_loc+'nullModel/v4/files/model/figure2/' #model sim files
	figuloc = root_loc+'nullModel/v4/figures/' #figure files

	#dataset short names, types, and colors

	datasets = { 'AcademicRanking' : 'universities', 'AtlasComplex' : 'countries', 'Citations' : 'scientists', 'Cities_RU' : 'cities (RU)', 'Cities_UK' : 'cities (GB)', 'Earthquakes_avgMagnitude' : 'regions JP\n(quake mag)', 'Earthquakes_numberQuakes' : 'regions JP\n(quakes)', 'english' : 'English', 'enron-sent-mails-weekly' : 'Enron emails', 'FIDEFemale' : 'chess players\n(female)', 'FIDEMale' : 'chess players\n(male)', 'Football_FIFA' : 'national football\nteams', 'Football_Scorers' : 'Football scorers', 'Fortune' : 'companies', 'french' : 'French', 'german' : 'German', 'github-watch-weekly' : 'GitHub\nrepositories', 'Golf_OWGR' : 'golf players', 'Hienas' : 'hyenas', 'italian' : 'Italian', 'metroMex' : 'metro stations\n(Mexico)', 'Nascar_BuschGrandNational' : 'Nascar drivers\n(Busch)', 'Nascar_WinstonCupGrandNational' : 'Nascar drivers\n(Winston Cup)', 'Poker_GPI' : 'poker players', 'russian' : 'Russian', 'spanish' : 'Spanish', 'Tennis_ATP' : 'tennis players', 'TheGuardian_avgRecommends' : 'The Guardian\nreaders (recc)', 'TheGuardian_numberComments' : 'The Guardian\nreaders (comm)', 'UndergroundByWeek' : 'metro stations\n(London)' } #name dict
#	datasets = { 'VideogameEarnings' : 'videogame\nplayers', 'Virus' : 'viruses' } #shady data

	datasets_oneliners = { 'AcademicRanking' : 'universities', 'AtlasComplex' : 'countries', 'Citations' : 'scientists', 'Cities_RU' : 'cities (RU)', 'Cities_UK' : 'cities (GB)', 'Earthquakes_avgMagnitude' : 'regions JP (quake mag)', 'Earthquakes_numberQuakes' : 'regions JP(quakes)', 'english' : 'English', 'enron-sent-mails-weekly' : 'Enron emails', 'FIDEFemale' : 'chess players (female)', 'FIDEMale' : 'chess players (male)', 'Football_FIFA' : 'national football teams', 'Football_Scorers' : 'Football scorers', 'Fortune' : 'companies', 'french' : 'French', 'german' : 'German', 'github-watch-weekly' : 'GitHub repositories', 'Golf_OWGR' : 'golf players', 'Hienas' : 'hyenas', 'italian' : 'Italian', 'metroMex' : 'metro stations (Mexico)', 'Nascar_BuschGrandNational' : 'Nascar drivers (Busch)', 'Nascar_WinstonCupGrandNational' : 'Nascar drivers (Winston Cup)', 'Poker_GPI' : 'poker players', 'russian' : 'Russian', 'spanish' : 'Spanish', 'Tennis_ATP' : 'tennis players', 'TheGuardian_avgRecommends' : 'The Guardian readers (recc)', 'TheGuardian_numberComments' : 'The Guardian readers (comm)', 'UndergroundByWeek' : 'metro stations (London)' } #name dict
#	datasets_oneliners = { 'VideogameEarnings' : 'videogame players', 'Virus' : 'viruses' } #shady data

	datasets_openclosed = { 'AcademicRanking' : 'open', 'AtlasComplex' : 'open', 'Citations' : 'open', 'Cities_RU' : 'open', 'Cities_UK' : 'closed', 'Earthquakes_avgMagnitude' : 'closed', 'Earthquakes_numberQuakes' : 'closed', 'english' : 'open', 'enron-sent-mails-weekly' : 'open', 'FIDEFemale' : 'open', 'FIDEMale' : 'open', 'Football_FIFA' : 'closed', 'Football_Scorers' : 'open', 'Fortune' : 'open', 'french' : 'open', 'german' : 'open', 'github-watch-weekly' : 'open', 'Golf_OWGR' : 'open', 'Hienas' : 'open', 'italian' : 'open', 'metroMex' : 'closed', 'Nascar_BuschGrandNational' : 'open', 'Nascar_WinstonCupGrandNational' : 'open', 'Poker_GPI' : 'open', 'russian' : 'open', 'spanish' : 'open','Tennis_ATP' : 'open', 'TheGuardian_avgRecommends' : 'open', 'TheGuardian_numberComments' : 'open', 'UndergroundByWeek' : 'closed' } #type dict
#	datasets = { 'VideogameEarnings' : 'open', 'Virus' : 'open' } #shady data

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
	'marker_size' : 3,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':11 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables

	fig_props = { 'fig_num' : 2,
	'fig_size' : [ 30.48, 15.24 ],
	'aspect_ratio' : (3, 3),
	'grid_params' : dict( left=0.055, bottom=0.07, right=0.985, top=0.935, wspace=0.3, hspace=0.6 ),
	'height_ratios' : [1, 1, 1],
	'savename' : 'fig2_bot' } #for bottom figure

	svg_props = { 'fig_size' : [ '30.48cm', '15.24cm' ],
	'dpi' : '300',
	'added_file' : 'drawings/fig2_top',
	'savename' : 'figure2' } #for whole figure


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


# B: Displacement probability vs. rank in sims/theo

	#model parameters

	ptau, pnu = 0.1, 0.2 #probabilities of displacement/replacement
	t_vals = [1, 5] #selected times

	N = 100 #system size
	T = 10 #observation period
	p0 = 0.8 #relative size of ranking list
	ntimes = 10000 #number of realizations

	r = 0.4 #selected rank

	dr = 1 / float( N ) #rank increment
	N0 = int( p0*N ) #get ranking size
	sel_rank = int( N*r ) - 1 #selected rank (integer)

	#store parameters in dict
	params = { 'ptau' : ptau, 'pnu' : pnu, 'N' : N, 'N0' : N0, 'T' : T, 'ntimes' : ntimes }
	params[ 'sel_rank' ] = sel_rank #for displacement-in-time properties

	sel_tuples = [ (0, 1), (0, 2) ] #plot positions

	colors = sns.color_palette( 'Set2', n_colors=3 ) #colors to plot

	for grid_pos, grid_tuple in enumerate( sel_tuples ): #loop through subplot

		#initialise subplot
		ax = plt.subplot( grid[ grid_tuple ] )
		sns.despine( ax=ax )
		plt.xlabel( r'$X/N_0$', size=plot_props['xylabel'], labelpad=0 )
		if grid_pos == 0:
			plt.ylabel( r'$P_{x, t}$', size=plot_props['xylabel'], labelpad=1 )

		#prepare model

		t = t_vals[ grid_pos ] #get selected time

		#Levi sea (w/ replacement probability)
		levi_sea_rep = np.exp( -pnu * t ) * dr * ( 1 - np.exp( -ptau * t ) )

		#plot plot!

		xplot = np.linspace( 1. / N0, 1, num=N0 ) #values of displaced rank X / N_0

		#model
		dispprops_model, = model_misc.model_props( ['disp'], params, loadflag, saveloc_model, saveflag )['disp']
		yplot_model = dispprops_model.iloc[ :, t ] #selected t only
		plt.semilogy( xplot, yplot_model, 'o', label='simulations', c=colors[2], ms=plot_props['marker_size'] )

		#theo
		yplot_theo = model_misc.disp_time_theo( r, t, params )
		plt.semilogy( xplot, yplot_theo, label='analytical', c='0.4', lw=plot_props['linewidth'] )

		#Levi sea (w/ replacement probability)
		plt.fill_between( xplot, levi_sea_rep, y2=1e-4, color=colors[1], lw=0, label=None, alpha=0.5 )

		#diffusion peak (w/ replacement probability)
		plt.fill_between( xplot, yplot_theo, y2=levi_sea_rep, color=colors[0], lw=0, label=None, alpha=0.5 )

		#line at x = r
		plt.vlines( r/p0, 1e-4, yplot_theo[sel_rank], ls='--', colors='0.6', label='$r = x$', lw=plot_props['linewidth'] )

		#equations and texts
		if grid_pos == 0:

			eq_str = r'$P_{x, t} = e^{-\nu t} ( L_t + D_{x, t} )$'
			plt.text( 1.1, 1.05, eq_str, va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['xylabel'] )

			eq_str = r'$\sum_x D_{x, t} = e^{-\tau t}$'
			plt.annotate( text=eq_str, xy=( 0.6, 0.45 ), xytext=( 0.73, 0.65 ), va='bottom', ha='center', arrowprops=dict( arrowstyle='simple', color=colors[0], alpha=0.5  ), xycoords=ax.transAxes, textcoords=ax.transAxes )

			eq_str = r'$\sum_x L_t = 1 - e^{-\tau t}$'
			plt.annotate( text=eq_str, xy=( 0.22, 0.25 ), xytext=( 0.22, 0.5 ), va='bottom', ha='center', arrowprops=dict( arrowstyle='simple', color=colors[1], alpha=0.5 ), xycoords=ax.transAxes, textcoords=ax.transAxes )

			arrow_str = 'time $t$'
			bbox_props = dict( boxstyle="rarrow,pad=0.3", fc='None', ec='0.6', lw=2 )
			plt.text( 1.05, 0.5, arrow_str, ha='left', va='center', transform=ax.transAxes,
            size=plot_props['ticklabel'], bbox=bbox_props )

		#legend
		if grid_pos == 1:
			plt.legend( loc='upper right', bbox_to_anchor=(1.02, 1.3), prop=plot_props['legend_prop'], handlelength=1.7, numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'] )

		#finalise subplot
		plt.axis([ 0, 1, 1e-4, 1e0 ])
		ax.tick_params( axis='both', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=2 )
		if grid_pos == 1:
			plt.yticks([])


# C: Time dependence of rank flux in open/closed systems

	print( 'FLUX' ) #to know where we stand

	sel_datasets = [ 'TheGuardian_numberComments', 'Nascar_BuschGrandNational', 'Citations', 'AcademicRanking', 'metroMex' ] #open/closed examples
	sel_labels = { 'TheGuardian_numberComments' : 'The Guardian\nreaders\n(comm)', 'Nascar_BuschGrandNational' : 'Nascar\ndrivers\n(Busch)', 'Citations' : 'scientists', 'AcademicRanking' : 'universities', 'metroMex' : 'metro\nstations\n(Mexico)' } #selected names for datasets

	#get colors
	colors = sns.color_palette( 'coolwarm', n_colors=len( sel_datasets ) )

	#initialise subplot
	subgrid = grid[ 1:, 0 ].subgridspec( 1, 2, wspace=0, width_ratios=[1, 0.8] )
	ax = plt.subplot( subgrid[ 0 ] )
	sns.despine( ax=ax )
	plt.xlabel( '$t / T$', size=plot_props['xylabel'], labelpad=0 )
	plt.ylabel( r'$F_t$', size=plot_props['xylabel'], labelpad=0 )

	for pos_dset, dataname in enumerate( sel_datasets ): #loop through selected datasets
		print( '\t'+dataname ) #to know where we stand

		#prepare data

		#get parameters for dataset
		params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

		#get flux in data
		fluIprops_data = pd.read_pickle( saveloc_data + 'fluIprops_' + param_str_data )
		flux_data = fluIprops_data.mean( axis=1 ) #IN-flow (=OUT-flow when averaged over Nt)

		#prepare model

		datatype = datasets_openclosed[ dataname ] #dataset kind: open, closed

		#get model parameters for selected dataset
		params_model = data_misc.data_estimate_params_all( dataname, params, loadflag, saveloc_data, datatype=datatype )
		params[ 'pnu' ], params[ 'ptau' ] = params_model.loc[ 'optimal', [ 'pnu', 'ptau' ] ] #set parameters

		print( 'ptau = {:.4f}, pnu = {:.4f}'.format( params['ptau'], params['pnu'] ) ) #to know where we stand

		flux_model = model_misc.flux_theo( params ) * np.ones(( T - 1 )) #constant flow over time

		#plot plot!

		xplot = np.arange( 1, T ) / T #normalised time = ( 1, ..., T-1 ) / T

		#data
		plt.plot( xplot, flux_data, c=colors[pos_dset], label=sel_labels[ dataname ], lw=plot_props['linewidth'] )

		#model
		label = 'model' if pos_dset == len(sel_datasets)-1 else None
		plt.plot( xplot, flux_model, '--', c=colors[pos_dset], label=label, lw=plot_props['linewidth'] )

	#legend
	leg = plt.legend( loc='lower left', bbox_to_anchor=(0.95, 0.1), prop=plot_props['legend_prop'], handlelength=1.7, numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=1 )

#	#texts
	plt.text( 1, 0.95, 'more open', va='center', ha='right', weight='bold', transform=ax.transAxes, fontsize=plot_props['ticklabel']-1 )
	plt.text( 1, 0.03, 'less open', va='center', ha='right', weight='bold', transform=ax.transAxes, fontsize=plot_props['ticklabel']-1 )

	#finalise subplot
	plt.axis([ -0.05, 1.05, -0.05, 0.8 ])
	ax.tick_params( axis='both', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=2 )
	ax.locator_params( nbins=4 ) #change number of ticks in axes



# D1: Displacement probability for open/closed systems

	print( 'DISPLACEMENT' ) #to know where we stand

	sel_datasets = [ 'AcademicRanking', 'Football_FIFA' ] #open/closed examples
	sel_tuples = [ 1, 2 ] #plot positions
	sel_pos = [ 2, 4, 6, 8 ] #selected positions of ranks to show
	filter_factor = { 'AcademicRanking' : 9, 'Football_FIFA' : 5 } #factor for plotting filter

	#get colors
	colors = sns.color_palette( 'coolwarm', n_colors=len( sel_pos ) )

	subgrid = grid[ 1:, 1 ].subgridspec( 3, 1, hspace=0.5, height_ratios=[0.01, 1, 1] )

	for grid_pos, grid_tuple in enumerate( sel_tuples ): #loop through subplot

		#initialise subplot
		ax = plt.subplot( subgrid[ grid_tuple ] )
		sns.despine( ax=ax )

		if grid_pos == 1:
			plt.xlabel( '$X / N_0$', size=plot_props['xylabel'], labelpad=1 )
		plt.ylabel( r'$P_{x, t}$', size=plot_props['xylabel'], labelpad=1 )

		#prepare data

		#select dataset
		dataname = sel_datasets[ grid_pos ]
		print( '\t'+dataname ) #to know where we stand

		#get parameters for dataset
		params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

		#get displacement probability in data
		dispprops_data = pd.read_pickle( saveloc_data + 'dispprops_' + param_str_data )

		#prepare model

		dr = 1 / float( N ) #rank increment

		datatype = datasets_openclosed[ dataname ] #dataset kind: open, closed

		#get model parameters for selected dataset
		params_model = data_misc.data_estimate_params_all( dataname, params, loadflag, saveloc_data, datatype=datatype )
		params[ 'pnu' ], params[ 'ptau' ] = params_model.loc[ 'optimal', [ 'pnu', 'ptau' ] ] #set parameters

		print( 'ptau = {:.4f}, pnu = {:.4f}'.format( params['ptau'], params['pnu'] ) ) #to know where we stand

		#plot plot!

		xplot = np.arange( 1, N0+1 ) / N0 #normalised rank = ( 1, ..., N0 ) / N0

		for pos_rank, rank in enumerate( dispprops_data.columns[ sel_pos ] ): #loop through selected ranks

			#for 1st legend
			label = '{0:.1f}'.format( (rank + 1) / float(N0) ) #labels for legend

			#unfiltered/filtered data

			yplot_data = dispprops_data[ rank ] #get displacement for r
			yplot_data_filter = ss.savgol_filter( yplot_data, filter_factor[dataname], 2 )

			line_data, = plt.plot( xplot, yplot_data_filter, label=label, c=colors[ pos_rank ], lw=plot_props['linewidth'], zorder=0 )

			#unfiltered model
			r = dr * ( rank + 1 ) #get r value
			yplot_model = model_misc.displacement_theo( r, params )
			line_model, = plt.plot( xplot, yplot_model, '--', c=colors[ pos_rank ], lw=plot_props['linewidth'], zorder=1 )

			#for 2nd legend
			if grid_pos == 1 and pos_rank == 0:
				leg_lines = ( line_data, line_model )
				leg_labels = ( 'data', 'model' )

		#texts
		plot_str = r'$\tau =$ '+'{0:.3f}'.format( params['ptau'] )+'\n'+r'$\nu =$ '+'{0:.3f}'.format( params['pnu'] )
		plt.text( 0.9, 0.75, plot_str, va='top', ha='center', transform=ax.transAxes, fontsize=10)

		if grid_pos == 0:
			plt.text( 0, 1.17, '$R / N_0 =$', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['legend_prop']['size'] )
			plt.text( 0.9, 1.04, datasets[ dataname ], va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )
		else:
			plt.text( 0.9, 1.25, datasets[ dataname ], va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		#arrows
		if grid_pos == 0:
			plt.annotate( text='', xy=( 0.25, 1 ), xytext=( 0.74, 0.67 ), arrowprops=dict(arrowstyle='<->', color='0.6', connectionstyle='arc3,rad=-0.1'), xycoords=ax.transAxes, textcoords=ax.transAxes )
		else:
			plt.annotate( text='', xy=( 0.25, 0.94 ), xytext=( 0.74, 0.94 ), arrowprops=dict(arrowstyle='<->', color='0.6', connectionstyle='arc3,rad=-0.2'), xycoords=ax.transAxes, textcoords=ax.transAxes )

		#legends
		if grid_pos == 0:
			leg = plt.legend( loc='lower left', bbox_to_anchor=(0.07, 1), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=len( sel_pos ) )
		else:
			leg = plt.legend( leg_lines, leg_labels, loc='lower left', bbox_to_anchor=(-0.1, 1), prop=plot_props['legend_prop'], handlelength=1.7, numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=2 )


		#finalise subplot
		plt.axis([ -0.05, 1.05, -0.02, 0.22 ])
		ax.tick_params( axis='both', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=2 )
		if grid_pos == 0:
			plt.xticks([])


# D2: Rank dependence of rank change in open/closed systems

	print( 'CHANGE' ) #to know where we stand

	sel_datasets = [ 'AtlasComplex', 'Fortune', 'Cities_UK', 'metroMex' ] #open/closed examples
	sel_tuples = [ (1, 0), (1, 1), (2, 0), (2, 1) ] #plot positions

	#get colors
	colors = sns.color_palette( 'coolwarm', n_colors=5 )

	subgrid = grid[ 1:, 2 ].subgridspec( 3, 3, hspace=0.7, wspace=0.25, height_ratios=[0.05, 1, 1], width_ratios=[1, 1, 0.05] )

	for grid_pos, grid_tuple in enumerate( sel_tuples ): #loop through subplot

		#initialise subplot
		ax = plt.subplot( subgrid[ grid_tuple ] )
		sns.despine( ax=ax )
		if grid_pos in [2, 3]:
			plt.xlabel( '$R / N_0$', size=plot_props['xylabel'], labelpad=1 )
		if grid_pos in [0, 2]:
			plt.ylabel( r'$C$', size=plot_props['xylabel'], labelpad=2 )

		#prepare data

		#select dataset
		dataname = sel_datasets[ grid_pos ]
		print( '\t'+dataname ) #to know where we stand

		#get parameters for dataset
		params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

		#get rank change in data
		rankprops_data = pd.read_pickle( saveloc_data + 'rankprops_' + param_str_data )
		rankchange_data = rankprops_data.loc[ 'rankchange' ]

		#prepare model

		datatype = datasets_openclosed[ dataname ] #dataset kind: open, closed

		#get model parameters for selected dataset
		params_model = data_misc.data_estimate_params_all( dataname, params, loadflag, saveloc_data, datatype=datatype )
		params[ 'pnu' ], params[ 'ptau' ] = params_model.loc[ 'optimal', [ 'pnu', 'ptau' ] ] #set parameters

		print( 'ptau = {:.4f}, pnu = {:.4f}'.format( params['ptau'], params['pnu'] ) ) #to know where we stand

		rankchange_model = model_misc.change_theo( params )

		#plot plot!

		xplot = np.arange( 1, N0+1 ) / N0 #normalised rank = ( 1, ..., N0 ) / N0

		col = colors[0] if grid_pos in [0, 1] else colors[-1] #color for open/closed systems

		#filtered data
		filter_win = int( np.ceil( (N0 / 50 ) / 2 ) * 2 ) + 1 #filter window len (odd int)
		yplot_data = ss.savgol_filter( rankchange_data, filter_win, 2 )
		plt.plot( xplot, yplot_data, c=col, label='data', lw=plot_props['linewidth'] )

		#model
		plt.plot( xplot, rankchange_model, '--', c=col, label='model', lw=plot_props['linewidth'] )

		#texts

		plt.text( 0.5, 1, datasets[ dataname ], va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		plot_str = r'$\tau =$ '+'{0:.3f}'.format( params['ptau'] )+'\n'+r'$\nu =$ '+'{0:.3f}'.format( params['pnu'] )
		plt.text( 0.5, 0.03, plot_str, va='bottom', ha='center', transform=ax.transAxes, fontsize=10)

		if grid_pos == 1:
			plt.text( 1.2, 0.5, 'more open', va='center', ha='center', transform=ax.transAxes, rotation='vertical', weight='bold', fontsize=plot_props['ticklabel'] )
		if grid_pos == 3:
			plt.text( 1.2, 0.5, 'less open', va='center', ha='center', transform=ax.transAxes, rotation='vertical', weight='bold', fontsize=plot_props['ticklabel'] )
			plt.annotate( text='', xy=( 1.2, 1.5 ), xytext=( 1.2, 0.95 ), arrowprops=dict(arrowstyle='<->', color='0.6'), xycoords=ax.transAxes, textcoords=ax.transAxes )

		#legends
		if grid_pos == 0:
			leg = plt.legend( loc='lower left', bbox_to_anchor=(0.5, 1.15), prop=plot_props['legend_prop'], handlelength=1.7, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=2 )

		#finalise subplot
		plt.axis([ -0.05, 1.05, -0.05, 1.05 ])
		ax.tick_params( axis='both', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=2 )
		if grid_pos in [0, 1]:
			plt.xticks([])
		if grid_pos in [1, 3]:
			plt.yticks([])
		ax.locator_params( nbins=4 ) #change number of ticks in axes

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
			  sc.Text( 'b', 290, 18, size=plot_props['figlabel']-5, weight='bold' ),
			  sc.Text( 'c', 3, 175, size=plot_props['figlabel']-5, weight='bold' ),
			  sc.Text( 'd', 290, 175, size=plot_props['figlabel']-5, weight='bold' )
#			  sc.Text( 'e', 590, 175, size=plot_props['figlabel']-5, weight='bold' )
		).scale( 0.0354 ),

		sc.Panel( sc.SVG( svg_props['added_file']+'.svg' ),
			  sc.Text( 'a', 3, 18, size=plot_props['figlabel'], weight='bold' )
		).scale( 0.0278 ),

		# sc.Grid( 20, 20 )

		).save( svg_props['savename']+'.svg' )

		#save final pdf figure
		os.system( 'inkscape --export-dpi='+svg_props['dpi']+' --export-filename='+figuloc+svg_props['savename']+'.pdf '+figuloc+svg_props['savename']+'.svg' )

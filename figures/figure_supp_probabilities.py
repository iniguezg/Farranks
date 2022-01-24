#! /usr/bin/env python

#Farranks - Exploring rank dynamics in complex systems
#Copyright (C) 2022 Gerardo Iñiguez

### SCRIPT FOR PLOTTING SUPP FIGURE (PROBABILITIES) IN FARRANKS PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser
from matplotlib.ticker import ( MultipleLocator, LogLocator )

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc, model_misc


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	#flags and locations
	loadflag = 'y'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_data = root_loc+'nullModel/v4/files/' #location of output files

	#dataset short names, types, and colors

	datasets = { 'AcademicRanking' : 'universities', 'AtlasComplex' : 'countries', 'Citations' : 'scientists', 'Cities_RU' : 'cities (RU)', 'Cities_UK' : 'cities (GB)', 'Earthquakes_avgMagnitude' : 'regions JP\n(quake mag)', 'Earthquakes_numberQuakes' : 'regions JP\n(quakes)', 'english' : 'English', 'enron-sent-mails-weekly' : 'Enron emails', 'FIDEFemale' : 'chess players\n(female)', 'FIDEMale' : 'chess players\n(male)', 'Football_FIFA' : 'national football\nteams', 'Football_Scorers' : 'Football scorers', 'Fortune' : 'companies', 'french' : 'French', 'german' : 'German', 'github-watch-weekly' : 'GitHub\nrepositories', 'Golf_OWGR' : 'golf players', 'Hienas' : 'hyenas', 'italian' : 'Italian', 'metroMex' : 'metro stations\n(Mexico)', 'Nascar_BuschGrandNational' : 'Nascar drivers\n(Busch)', 'Nascar_WinstonCupGrandNational' : 'Nascar drivers\n(Winston Cup)', 'Poker_GPI' : 'poker players', 'russian' : 'Russian', 'spanish' : 'Spanish', 'Tennis_ATP' : 'tennis players', 'TheGuardian_avgRecommends' : 'The Guardian\nreaders (recc)', 'TheGuardian_numberComments' : 'The Guardian\nreaders (comm)', 'UndergroundByWeek' : 'metro stations\n(London)' } #name dict

	datasets_openclosed = { 'AcademicRanking' : 'open', 'AtlasComplex' : 'open', 'Citations' : 'open', 'Cities_RU' : 'open', 'Cities_UK' : 'closed', 'Earthquakes_avgMagnitude' : 'closed', 'Earthquakes_numberQuakes' : 'closed', 'english' : 'open', 'enron-sent-mails-weekly' : 'open', 'FIDEFemale' : 'open', 'FIDEMale' : 'open', 'Football_FIFA' : 'closed', 'Football_Scorers' : 'open', 'Fortune' : 'open', 'french' : 'open', 'german' : 'open', 'github-watch-weekly' : 'open', 'Golf_OWGR' : 'open', 'Hienas' : 'open', 'italian' : 'open', 'metroMex' : 'closed', 'Nascar_BuschGrandNational' : 'open', 'Nascar_WinstonCupGrandNational' : 'open', 'Poker_GPI' : 'open', 'russian' : 'open', 'spanish' : 'open','Tennis_ATP' : 'open', 'TheGuardian_avgRecommends' : 'open', 'TheGuardian_numberComments' : 'open', 'UndergroundByWeek' : 'closed' } #type dict

	datatypes = { 'AcademicRanking' : 'society', 'AtlasComplex' : 'economics', 'Citations' : 'society', 'Cities_RU' : 'infrastructure', 'Cities_UK' : 'infrastructure', 'Earthquakes_avgMagnitude' : 'nature', 'Earthquakes_numberQuakes' : 'nature', 'english' : 'languages', 'enron-sent-mails-weekly' : 'society', 'FIDEFemale' : 'sports', 'FIDEMale' : 'sports', 'Football_FIFA' : 'sports', 'Football_Scorers' : 'sports', 'Fortune' : 'economics', 'french' : 'languages', 'german' : 'languages', 'github-watch-weekly' : 'society', 'Golf_OWGR' : 'sports', 'Hienas' : 'nature', 'italian' : 'languages', 'metroMex' : 'infrastructure', 'Nascar_BuschGrandNational' : 'sports', 'Nascar_WinstonCupGrandNational' : 'sports', 'Poker_GPI' : 'sports', 'russian' : 'languages', 'spanish' : 'languages','Tennis_ATP' : 'sports', 'TheGuardian_avgRecommends' : 'society', 'TheGuardian_numberComments' : 'society', 'UndergroundByWeek' : 'infrastructure' } #type dict

	palette = sns.color_palette( 'Set2', n_colors=7 ) #selected colormap for types
	datacols = { 'society' : palette[0], 'languages' : palette[1], 'economics' : palette[2], 'infrastructure' : palette[3], 'nature' : palette[4], 'sports' : palette[6] } #set color for dataset type

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 13,
	'text_size' : 10,
	'marker_size' : 6,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':10 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (10, 6),
	'aspect_ratio' : (4, 6),
	'grid_params' : dict( left=0.085, bottom=0.105, right=0.985, top=0.87, wspace=0.4, hspace=0.7 ),
	'dpi' : 300,
	'savename' : 'figure_supp_probabilities' }

	#get colors
	colors = sns.color_palette( 'Paired', n_colors=3 ) #colors to plot


	## DATA ##

	#get parameters for all datasets
	params_data = pd.read_pickle( saveloc_data+'params_data.pkl' )

	#prepare dataset order by flux

	fluxmean_data = pd.Series( np.zeros( len(datasets) ), index=datasets.keys(), name='flux_mean' )
	for dataname in datasets: #loop through datasets

		#get parameters for dataset
		params = params_data.loc[ dataname ]
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		param_str = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

		#get flux mean in data
		fluOprops_data = pd.read_pickle( saveloc_data + 'fluOprops_' + param_str )
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


# A: Regimes of Levy/Diffusion/replacement dynamics in open/closed systems

	sel_datasets = [ dataname for dataname in fluxmean_data.index if datasets_openclosed[ dataname ] == 'open' ]

	for grid_pos, dataname in enumerate( sel_datasets ): #loop through (open!) datasets (in order by decreasing mean flux)
		print( 'flux = {:.2f}, dataset = {}'.format( fluxmean_data[ dataname ], dataname ) ) #to know where we stand

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos ] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [ 18, 19, 20, 21, 22, 23 ]:
			plt.xlabel( r'$\nu_r$', size=plot_props['xylabel'], labelpad=2 )
		if grid_pos in [ 0, 6, 12, 18 ]:
			plt.ylabel( r'$W_{\bullet}$', size=plot_props['xylabel'], labelpad=2 )

		#prepare data

		params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

		p0 = N0 / float( N ) #ranking fraction

		#average openness derivative in data
		openprops_data = pd.read_pickle( saveloc_data + 'openprops_' + param_str_data )
		open_deriv = openprops_data.loc[ 'open_deriv' ].mean() #get mean

		#get model parameters for selected dataset
		params_model = data_misc.data_estimate_params_all( dataname, params, loadflag, saveloc_data, datatype='open' )
		pnu, ptau = params_model.loc[ 'optimal', [ 'pnu', 'ptau' ] ] #set parameters

		print( 'ptau = {:.4f}, pnu = {:.4f}'.format( ptau, pnu ) ) #to know where we stand

		#get rescaled model parameters
		pnu_resc = ( pnu - p0 * open_deriv ) / open_deriv
		ptau_resc = ptau / ( p0 * (1 - p0) * open_deriv )

		#regime probabilities (with t=1)
		W_repl = 1 - np.exp( -pnu )
		W_diff = np.exp( -pnu ) * np.exp( -ptau )
		W_levy = np.exp( -pnu ) * ( 1 - np.exp( -ptau ) )

		#prepare (variable) model parameters

		#rescaled parameters
		pnu_resc_vals = np.logspace( -3, np.log10( 1 - p0 ), 50 ) #pick rescaled pnu as variable
		ptau_resc_vals = 1 / pnu_resc_vals #and slide over universal curve

		#model parameters
		pnu_vals = ( pnu_resc_vals + p0 ) * open_deriv
		ptau_vals = ptau_resc_vals * p0 * (1 - p0) * open_deriv

		#regime probabilities (with t=1)
		W_repl_vals = 1 - np.exp( -pnu_vals )
		W_diff_vals = np.exp( -pnu_vals ) * np.exp( -ptau_vals )
		W_levy_vals = np.exp( -pnu_vals ) * ( 1 - np.exp( -ptau_vals ) )


		#plot plot!

		#regime probabilities (in data)
		handle_data, = plt.semilogx( pnu_resc, W_levy, 'o', label=None, ms=plot_props['marker_size'], c=colors[0], zorder=2 )
		plt.semilogx( pnu_resc, W_diff, 'o', label=None, ms=plot_props['marker_size'], c=colors[1], zorder=2 )
		plt.semilogx( pnu_resc, W_repl, 'o', label=None, ms=plot_props['marker_size'], c=colors[2], zorder=2 )

		#(variable) regime probabilities
		handle_model, = plt.semilogx( pnu_resc_vals, W_levy_vals, label='$W_{\mathrm{levy}}$', lw=plot_props['linewidth'], c=colors[0], zorder=1 )
		plt.semilogx( pnu_resc_vals, W_diff_vals, label='$W_{\mathrm{diff}}$', lw=plot_props['linewidth'], c=colors[1], zorder=1 )
		plt.semilogx( pnu_resc_vals, W_repl_vals, label='$W_{\mathrm{repl}}$', lw=plot_props['linewidth'], c=colors[2], zorder=1 )

		#line at rescaled pnu
		handle_param = plt.axvline( pnu_resc, ls='--', c='0.6', label=None, lw=plot_props['linewidth']-1, zorder=0 )

		#texts
		plt.text( 0.5, 1.3, datasets[ dataname ], va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#mean flux
		fluxmean_str = r'$F =$ '+'{0:.2f}'.format( fluxmean_data[ dataname ] )
		plt.text( 0.99, 0.5, fluxmean_str, va='center', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'])

		#legends
		if grid_pos == 0:
			leg1 = plt.legend( loc='lower left', bbox_to_anchor=(1.7, 1.5), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=3 )
			ax.add_artist(leg1)

			leg2 = plt.legend( ( handle_data, handle_model, handle_param ), ( 'data', 'model', r'$\nu^*$' ), loc='lower left', bbox_to_anchor=(4.3, 1.5), prop=plot_props['legend_prop'], handlelength=1.7, numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=3 )
			ax.add_artist(leg2)

		#finalise subplot
		plt.axis([ 1e-3, 1e-0, 0, 1 ])
		ax.xaxis.set_major_locator( LogLocator( numticks=4 ) )
		ax.locator_params( axis='y', nbins=3 )
		ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )
		if grid_pos not in [ 18, 19, 20, 21, 22, 23 ]:
			plt.xticks([])
		if grid_pos not in [ 0, 6, 12, 18 ]:
			plt.yticks([])


	#texts and arrows
	plt.text( -7.65, 5.5, 'open', va='center', ha='center', transform=ax.transAxes, weight='bold', rotation='vertical', fontsize=plot_props['xylabel'] )
	plt.annotate( text='', xy=( -7.65, 0.3 ), xytext=( -7.65, 5 ), arrowprops=dict(arrowstyle='<-', color='0.6'), xycoords=ax.transAxes, textcoords=ax.transAxes )

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )

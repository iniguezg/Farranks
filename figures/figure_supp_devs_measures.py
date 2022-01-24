#! /usr/bin/env python

#Farranks - Exploring rank dynamics in complex systems
#Copyright (C) 2022 Gerardo Iñiguez

### SCRIPT FOR PLOTTING SUPP FIGURE (DEVIATIONS IN MEASURES) IN FARRANKS PROJECT ###

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
	saveloc_data = root_loc+'nullModel/v4/files/' #location of original/sampling files

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
	'grid_params' : dict( left=0.08, bottom=0.08, right=0.995, top=0.88, wspace=0.3, hspace=0.6 ),
	'dpi' : 300,
	'savename' : 'figure_supp_devs_measures' }

	colors = sns.color_palette( 'Set2', n_colors=3 )


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


# A: Deviations in flux/open/succ measures for bootstrapped model

	sel_datasets = [ dataname for dataname in fluxmean_data.index if datasets_openclosed[ dataname ] == 'open' ]

	for grid_pos, dataname in enumerate( sel_datasets ): #loop through (open!) datasets (in order by decreasing mean flux)
		print( 'flux = {:.2f}, dataset = {}'.format( fluxmean_data[ dataname ], dataname ) ) #to know where we stand

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos ] )
		sns.despine( ax=ax ) #take out spines

		#prepare data

		#load fitted model parameters and measures in data
		params_model = pd.read_pickle( saveloc_data+'params_model_'+dataname+'.pkl' )
		flux = params_model.loc['optimal', 'flux']
		open_deriv = params_model.loc['optimal', 'open_deriv']
		success = params_model.loc['optimal', 'success']

		#load parameters/measures of bootstrapped samples of model
		params_devs = pd.read_pickle( saveloc_data+'params_devs_'+dataname+'.pkl' )
		#NOTE: for incomplete sampling, just keep finished realisations
		params_devs = params_devs[ params_devs.p0 > 0 ]
		print( '\trealisations: {}'.format( params_devs.index.size ) )

		#statistical significance of deviations

		#rescale bootstrapped measures relative to original values
		flux_frac = ( params_devs.flux - flux ).rename('flux_frac')
		open_deriv_frac = ( params_devs.open_deriv - open_deriv ).rename('open_deriv_frac')
		success_frac = ( params_devs.success - success ).rename('success_frac')
		data = pd.concat( [ flux_frac, open_deriv_frac, success_frac ], axis=1 )


		#plot plot!

		#KDEs of bootstrapped model
		sns.histplot( data=data, x='flux_frac', label='$F_{\mathrm{sim}} - F$', kde=True, ax=ax, element='step', color=colors[0], zorder=0 )
		sns.histplot( data=data, x='open_deriv_frac', label=r'$\dot{o}_{\mathrm{sim}} - \dot{o}$', kde=True, ax=ax, element='step', color=colors[1], zorder=0 )
		sns.histplot( data=data, x='success_frac', label=r'$S^{++}_{\mathrm{sim}} - S^{++}$', kde=True, ax=ax, element='step', color=colors[2], zorder=0 )

		#reference value for data
		plt.axvline( x=0, ls='--', c='0.5', lw=plot_props['linewidth'], zorder=1 )

		#texts
		plt.text( 0.5, 1.2, datasets[ dataname ], va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#legend
		if grid_pos == 0:
			leg = plt.legend( loc='lower left', bbox_to_anchor=(2.5, 1.43), prop=plot_props['legend_prop'], handlelength=1.7, numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=3 )

		if grid_pos == 0:
			plt.text( -0.59, 0.5, 'open', va='center', ha='center', transform=ax.transAxes, weight='bold', rotation='vertical', fontsize=plot_props['xylabel'] )
			plt.annotate( text='', xy=( -0.59, -4.5 ), xytext=( -0.59, 0 ), arrowprops=dict(arrowstyle='<-', color='0.6'), xycoords=ax.transAxes, textcoords=ax.transAxes )

		#finalise subplot
		if grid_pos in [ 18, 19, 20, 21, 22, 23 ]:
			plt.xlabel( r'$\bullet$', size=plot_props['xylabel'], labelpad=2 )
		else:
			plt.xlabel('')
			plt.xticks([])
		if grid_pos in [ 0, 6, 12, 18 ]:
			plt.ylabel( r'count', size=plot_props['xylabel'], labelpad=2 )
		else:
			plt.ylabel('')
		lim_val = 0.1
		ax.set_xlim( -1.2*lim_val, 1.2*lim_val )
		ax.locator_params( axis='both', nbins=3 )
		ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )

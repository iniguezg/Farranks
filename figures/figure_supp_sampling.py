#! /usr/bin/env python

#Farranks - Exploring rank dynamics in complex systems
#Copyright (C) 2022 Gerardo Iñiguez

### SCRIPT FOR PLOTTING SUPP FIGURE (SAMPLING) IN FARRANKS PROJECT ###

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
	saveloc_orig = root_loc+'nullModel/v4/files/' #location of original/sampling files
	saveloc_samp = root_loc+'nullModel/v4/files/sampling/'

	samp_thres = 0.1 #sampling threshold

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
	'marker_size' : 10,
	'linewidth' : 1,
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
	'aspect_ratio' : (6, 6),
	'grid_params' : dict( left=0.065, bottom=0.1, right=0.99, top=0.99, wspace=0.2, hspace=0.8 ),
	'height_ratios' : [ 0.1, 0, 1, 1, 1, 1 ],
	'dpi' : 300,
	'savename' : 'figure_supp_sampling' }


	## DATA ##

	#get parameters for all datasets
	params_data = pd.read_pickle( saveloc_orig+'params_data.pkl' )

	#prepare dataset order by flux

	fluxmean_data = pd.Series( np.zeros( len(datasets) ), index=datasets.keys(), name='flux_mean' )
	for dataname in datasets: #loop through datasets

		#get parameters for dataset
		params = params_data.loc[ dataname ]
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		param_str = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

		#get flux mean in data
		fluOprops_data = pd.read_pickle( saveloc_orig + 'fluOprops_' + param_str )
		#average OUT-/IN- flux over time, and then over ranks
		fluxmean_data[ dataname ] = fluOprops_data.mean( axis=0 ).mean()

	fluxmean_data.sort_values( ascending=False, inplace=True ) #sort values

	#universal curve in model

	pnu_resc_vals = np.logspace( -3, np.log10( 4e-1 ), 50 ) #pick rescaled pnu as variable
	ptau_resc_vals = 1 / pnu_resc_vals #and slide over universal curve


	## PLOTTING ##

	#initialise plot
	sns.set( style="white" ) #set fancy fancy plot
	#convert fig size to inches
	fig = plt.figure( fig_props['fig_num'], figsize=np.array( fig_props['fig_size'] ) )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'], height_ratios=fig_props['height_ratios'] )
	grid.update( **fig_props['grid_params'] )


# A: Rescaled parameters across universal curve under sampling

	sel_datasets = [ dataname for dataname in fluxmean_data.index if datasets_openclosed[ dataname ] == 'open' ]
	# sel_datasets = [ 'Earthquakes_numberQuakes' ]

	for grid_pos, dataname in enumerate( sel_datasets ): #loop through (open!) datasets (in order by decreasing mean flux)
		print( 'flux = {:.2f}, dataset = {}'.format( fluxmean_data[ dataname ], dataname ) ) #to know where we stand

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos + 12 ] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [ 18, 19, 20, 21, 22, 23 ]:
			plt.xlabel( r'$\nu_r$', size=plot_props['xylabel'], labelpad=2 )
		if grid_pos in [ 0, 6, 12, 18 ]:
			plt.ylabel( r'$\tau_r$', size=plot_props['xylabel'], labelpad=2 )

		#prepare data

		#load parameters and fitted model parameters for samples of original dataset
		params = params_data.loc[ dataname ] #original parameters
		params_sample = pd.read_pickle( saveloc_samp+'params_sample_'+dataname+'.pkl' )
		params_model = pd.read_pickle( saveloc_samp+'params_model_'+dataname+'.pkl' )
		params_all = pd.concat( [ params_sample, params_model ], axis=1 ) #join all

		#get rescaled model parameters (as a function of sampling jump)
		pnu_resc = ( ( params_all['pnu'] - params_all['p0'] * params_all['open_deriv'] ) / params_all['open_deriv'] ).rename('pnu_resc')
		ptau_resc = ( params_all['ptau'] / ( params_all['p0'] * (1 - params_all['p0']) * params_all['open_deriv'] ) ).rename('ptau_resc')
		#fraction of observations (out of T) left by (sub)sampling
		samp_frac = ( params_all['T'] / float( params['T'] ) ).rename('samp_frac')

		params_all = pd.concat( [ params_all, ptau_resc, pnu_resc, samp_frac ], axis=1 ) #join all

		#filter data (with sampling fraction larger than threshold!)
		params_plot = params_all[ params_all['samp_frac'] > samp_thres ]
		print(params_plot)

		#plot plot!

		#(filtered) sampled dataset
		scat = plt.scatter( params_plot['pnu_resc'], params_plot['ptau_resc'], c=params_plot['samp_frac'], vmin=0, vmax=1, cmap='winter_r', s=plot_props['marker_size'], zorder=1 )

		#universal curve in model
		plt.plot( pnu_resc_vals, ptau_resc_vals, '--', c='0.5', lw=plot_props['linewidth'], label=r'$\tau_r \nu_r = 1$', zorder=0 )

		#texts
		plt.text( 0.5, 1.15, datasets[ dataname ], va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#finalise subplot
		plt.axis([ 1e-3, 4e-1, 2.5e-0, 1e3 ])
		plt.xscale('log')
		plt.yscale('log')
		ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )
		if grid_pos not in [ 18, 19, 20, 21, 22, 23 ]:
			plt.xticks([])
		if grid_pos not in [ 0, 6, 12, 18 ]:
			plt.yticks([])

	#colorbar
	cax = plt.subplot( grid[ 2:4 ] )
	cbar = plt.colorbar( scat, cax=cax, orientation='horizontal' )
	plt.text( -0.15, -2.6, '$T_{\mathrm{eff}} / T=$', va='center', ha='center', transform=cax.transAxes, fontsize=plot_props['text_size'] )

	#texts and arrows
	plt.text( -7.65, 5.5, 'open', va='center', ha='center', transform=ax.transAxes, weight='bold', rotation='vertical', fontsize=plot_props['xylabel'] )
	plt.annotate( text='', xy=( -7.65, 0.3 ), xytext=( -7.65, 5 ), arrowprops=dict(arrowstyle='<-', color='0.6'), xycoords=ax.transAxes, textcoords=ax.transAxes )

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )

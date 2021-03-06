#! /usr/bin/env python

#Farranks - Exploring rank dynamics in complex systems
#Copyright (C) 2022 Gerardo Iñiguez

### SCRIPT FOR PLOTTING FIGURE 3 IN FARRANKS PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc, model_misc


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	#flags and locations
	loadflag = 'y'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_data = root_loc+'nullModel/v4/files/' #location of data/sampling files
	saveloc_samp = root_loc+'nullModel/v4/files/sampling/'
	figuloc = root_loc+'nullModel/v4/figures/' #location of figure files

	#dataset short names, types, and colors

	datasets = { 'AcademicRanking' : 'universities', 'AtlasComplex' : 'countries', 'Citations' : 'scientists', 'Cities_RU' : 'cities (RU)', 'Cities_UK' : 'cities (GB)', 'Earthquakes_avgMagnitude' : 'regions JP\n(quake mag)', 'Earthquakes_numberQuakes' : 'regions JP\n(quakes)', 'english' : 'English', 'enron-sent-mails-weekly' : 'Enron emails', 'FIDEFemale' : 'chess players\n(female)', 'FIDEMale' : 'chess players\n(male)', 'Football_FIFA' : 'national football\nteams', 'Football_Scorers' : 'Football scorers', 'Fortune' : 'companies', 'french' : 'French', 'german' : 'German', 'github-watch-weekly' : 'GitHub\nrepositories', 'Golf_OWGR' : 'golf players', 'Hienas' : 'hyenas', 'italian' : 'Italian', 'metroMex' : 'metro stations\n(Mexico)', 'Nascar_BuschGrandNational' : 'Nascar drivers\n(Busch)', 'Nascar_WinstonCupGrandNational' : 'Nascar drivers\n(Winston Cup)', 'Poker_GPI' : 'poker players', 'russian' : 'Russian', 'spanish' : 'Spanish', 'Tennis_ATP' : 'tennis players', 'TheGuardian_avgRecommends' : 'The Guardian\nreaders (recc)', 'TheGuardian_numberComments' : 'The Guardian\nreaders (comm)', 'UndergroundByWeek' : 'metro stations\n(London)' } #name dict

	datasets_oneliners = { 'AcademicRanking' : 'universities', 'AtlasComplex' : 'countries', 'Citations' : 'scientists', 'Cities_RU' : 'cities (RU)', 'Cities_UK' : 'cities (GB)', 'Earthquakes_avgMagnitude' : 'regions JP (quake mag)', 'Earthquakes_numberQuakes' : 'regions JP(quakes)', 'english' : 'English', 'enron-sent-mails-weekly' : 'Enron emails *', 'FIDEFemale' : 'chess players (female)', 'FIDEMale' : 'chess players (male)', 'Football_FIFA' : 'national football teams', 'Football_Scorers' : 'Football scorers', 'Fortune' : 'companies', 'french' : 'French', 'german' : 'German', 'github-watch-weekly' : 'GitHub repositories', 'Golf_OWGR' : 'golf players', 'Hienas' : 'hyenas *', 'italian' : 'Italian', 'metroMex' : 'metro stations (Mexico)', 'Nascar_BuschGrandNational' : 'Nascar drivers (Busch) *', 'Nascar_WinstonCupGrandNational' : 'Nascar drivers (Winston Cup)', 'Poker_GPI' : 'poker players', 'russian' : 'Russian', 'spanish' : 'Spanish', 'Tennis_ATP' : 'tennis players', 'TheGuardian_avgRecommends' : 'The Guardian readers (recc) *', 'TheGuardian_numberComments' : 'The Guardian readers (comm)', 'UndergroundByWeek' : 'metro stations (London)' } #name dict

	datasets_openclosed = { 'AcademicRanking' : 'open', 'AtlasComplex' : 'open', 'Citations' : 'open', 'Cities_RU' : 'open', 'Cities_UK' : 'closed', 'Earthquakes_avgMagnitude' : 'closed', 'Earthquakes_numberQuakes' : 'closed', 'english' : 'open', 'enron-sent-mails-weekly' : 'open', 'FIDEFemale' : 'open', 'FIDEMale' : 'open', 'Football_FIFA' : 'closed', 'Football_Scorers' : 'open', 'Fortune' : 'open', 'french' : 'open', 'german' : 'open', 'github-watch-weekly' : 'open', 'Golf_OWGR' : 'open', 'Hienas' : 'open', 'italian' : 'open', 'metroMex' : 'closed', 'Nascar_BuschGrandNational' : 'open', 'Nascar_WinstonCupGrandNational' : 'open', 'Poker_GPI' : 'open', 'russian' : 'open', 'spanish' : 'open','Tennis_ATP' : 'open', 'TheGuardian_avgRecommends' : 'open', 'TheGuardian_numberComments' : 'open', 'UndergroundByWeek' : 'closed' } #type dict

	datatypes = { 'AcademicRanking' : 'society', 'AtlasComplex' : 'economics', 'Citations' : 'society', 'Cities_RU' : 'infrastructure', 'Cities_UK' : 'infrastructure', 'Earthquakes_avgMagnitude' : 'nature', 'Earthquakes_numberQuakes' : 'nature', 'english' : 'languages', 'enron-sent-mails-weekly' : 'society', 'FIDEFemale' : 'sports', 'FIDEMale' : 'sports', 'Football_FIFA' : 'sports', 'Football_Scorers' : 'sports', 'Fortune' : 'economics', 'french' : 'languages', 'german' : 'languages', 'github-watch-weekly' : 'society', 'Golf_OWGR' : 'sports', 'Hienas' : 'nature', 'italian' : 'languages', 'metroMex' : 'infrastructure', 'Nascar_BuschGrandNational' : 'sports', 'Nascar_WinstonCupGrandNational' : 'sports', 'Poker_GPI' : 'sports', 'russian' : 'languages', 'spanish' : 'languages','Tennis_ATP' : 'sports', 'TheGuardian_avgRecommends' : 'society', 'TheGuardian_numberComments' : 'society', 'UndergroundByWeek' : 'infrastructure' } #type dict

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
	'legend_prop' : { 'size':10 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables

	fig_props = { 'fig_num' : 3,
	'fig_size' : (12, 6),
	'aspect_ratio' : (2, 2),
	'grid_params' : dict( left=0.045, bottom=0.07, right=0.985, top=0.98, wspace=0.2, hspace=0.6 ),
	'height_ratios' : [1, 0.3],
	'width_ratios' : [1, 0.3],
	'dpi' : 300,
	'savename' : 'figure3' }


	## DATA ##

	#get parameters for all datasets
	params_data = pd.read_pickle( saveloc_data+'params_data.pkl' )
	intervals_data = pd.read_pickle( saveloc_data+'intervals_data.pkl' )


	## PLOTTING ##

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=np.array( fig_props['fig_size'] ) )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'], height_ratios=fig_props['height_ratios'], width_ratios=fig_props['width_ratios'] )
	grid.update( **fig_props['grid_params'] )


# A: Universal curve of rescaled model parameters for open systems

	print( 'UNIVERSAL CURVE' ) #to know where we stand

	#label formats: x offset, y offset, va, ha
	formats = {
	'github-watch-weekly' : ( 20, 0, 'center', 'left' ),
	'TheGuardian_numberComments' : ( 20, 0, 'center', 'left' ),
	'TheGuardian_avgRecommends' : ( -15, -10, 'top', 'center' ),
	'french' : ( 33, 6, 'bottom', 'center' ),
	'enron-sent-mails-weekly' : ( 60, 13, 'center', 'center' ),
	'english' : ( -55, 8, 'top', 'right' ),
	'spanish' : ( 27, 7, 'center', 'left' ),
	'AtlasComplex' : ( -20, -8, 'bottom', 'right' ),
	'german' : ( 65, 0, 'center', 'left' ),
	'italian' : ( -20, -13, 'center', 'right' ),
	'russian' : ( 20, -1, 'top', 'left' ),
	'FIDEMale' : ( -20, -14, 'center', 'right' ),
	'Citations' : ( 20, 0, 'center', 'left' ),
	'Nascar_BuschGrandNational' : ( 20, 0, 'center', 'left' ),
	'FIDEFemale' : ( -30, 10, 'center', 'right' ),
	'AcademicRanking' : ( -25, 1, 'center', 'right' ),
	'Poker_GPI' : ( 20, 10, 'top', 'left' ),
	'Tennis_ATP' : ( -25, -8, 'bottom', 'right' ),
	'Golf_OWGR' : ( -20, -5, 'top', 'right' ),
	'Nascar_WinstonCupGrandNational' : ( -25, -13, 'top', 'right' ),
	'Football_Scorers' : ( 25, 2, 'center', 'left' ),
	'Hienas' : ( 20, -2, 'top', 'left' ),
	'Fortune' : ( -30, -11, 'center', 'right' ),
	'Cities_RU' : ( -20, -13, 'center', 'right' ) }

	#initialise subplot
	subgrid = grid[ 0, 0 ].subgridspec( 2, 1, hspace=0, height_ratios=[0.15, 1] )
	ax = plt.subplot( subgrid[ 1 ] )
	sns.despine( ax=ax ) #take out top and right spines
	plt.xlabel( r'$\nu_r$', size=plot_props['xylabel'], labelpad=-3 )
	plt.ylabel( r'$\tau_r$', size=plot_props['xylabel'], labelpad=0 )

	plt.text( -0.04, 1.05, 'a', va='bottom', ha='right', transform=ax.transAxes, fontsize=plot_props['figlabel'], weight='bold' )

	for type in [ 'society', 'languages', 'economics', 'infrastructure', 'nature', 'sports' ]: #loop through types (ordered like Table S1)
		sel_datasets = [ dataname for dataname in systems[ type ] if datasets_openclosed[ dataname ] == 'open' ] #consider only open datasets

		for pos, dataname in enumerate( sel_datasets ): #loop through (open) datasets
			print( 'dataset name: ' + dataname ) #print dataset

			#prepare data

			#get parameters for dataset
			params = params_data.loc[ dataname ]
			N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
			param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

			p0 = N0 / float( N ) #ranking fraction

			#mean flux in data
			fluOprops_data = pd.read_pickle( saveloc_data + 'fluOprops_' + param_str_data )
			flux_data = fluOprops_data.mean( axis=0 ).mean() #mean flux

			#average openness derivative in data
			openprops_data = pd.read_pickle( saveloc_data + 'openprops_' + param_str_data )
			open_deriv_data = openprops_data.loc[ 'open_deriv' ].mean() #get mean

			#get model parameters for selected dataset
			params_model = data_misc.data_estimate_params_all( dataname, params, loadflag, saveloc_data, datatype='open' )
			pnu, ptau = params_model.loc[ 'optimal', [ 'pnu', 'ptau' ] ] #set parameters

			#get rescaled model parameters
			pnu_resc = ( pnu - p0 * open_deriv_data ) / open_deriv_data
			ptau_resc = ptau / ( p0 * (1 - p0) * open_deriv_data )

			#plot plot!

			#parameter fits per dataset
			label = type if pos == 0 else '_nolegend_'
			plt.loglog( pnu_resc, ptau_resc, 'o', label=label, c=datacols[ type ], ms=plot_props['marker_size'], zorder=1 )

			#dataset names
			plt.annotate( text=datasets_oneliners[ dataname ], xy=( pnu_resc, ptau_resc ), xytext=( formats[dataname][0], formats[dataname][1]), va=formats[dataname][2], ha=formats[dataname][3], arrowprops=dict( headlength=1, headwidth=1, width=0.5, color=datacols[ datatypes[ dataname ] ] ), textcoords='offset points', size=plot_props['ticklabel']-1, color='0.4', zorder=1 )

	#universal curve in model
	pnu_resc_vals = np.logspace( -3, np.log10( 4e-1 ), 50 ) #pick rescaled pnu as variable
	ptau_resc_vals = 1 / pnu_resc_vals #and slide over universal curve
	plt.loglog( pnu_resc_vals, ptau_resc_vals, '--', c='0.5', lw=plot_props['linewidth'], label=r'$\tau_r \nu_r = 1$', zorder=0 )

	#legend
	leg = plt.legend( loc='lower left', bbox_to_anchor=(-0.02, 1.04), prop=plot_props['legend_prop'], handlelength=1.7, numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=len(systems)+1 )

	#finalise subplot
	plt.axis([ 1e-3, 4e-1, 2.5e-0, 1e3 ])
	ax.tick_params( axis='x', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=3 )
	ax.tick_params( axis='y', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=2 )
	plt.minorticks_off()


# B1: Effect of subsampling in open systems

	print( 'EFFECT OF SUBSAMPLING 1' ) #to know where we stand

	kmax = 15 #max sampling jump

	#initialise subplot
	subgrid = grid[ 0, 1 ].subgridspec( 4, 4, hspace=0.1, wspace=0, height_ratios=[1, 0.3, 0.05, 0.8] )
	ax = plt.subplot( subgrid[ 0, : ] )
	sns.despine( ax=ax ) #take out top and right spines
	plt.xlabel( r'$k$', size=plot_props['xylabel'], labelpad=-3 )
	ylabel_str = r'$\nu / k \ell$'+'\n(days$^{-1}$)'
	plt.ylabel( ylabel_str, size=plot_props['xylabel'], labelpad=0 )

	plt.text( -0.35, 0.8, 'b', va='bottom', ha='right', transform=ax.transAxes, fontsize=plot_props['figlabel'], weight='bold' )

	for type in [ 'society', 'languages', 'economics', 'infrastructure', 'nature', 'sports' ]: #loop through types (ordered like Table S1)
		sel_datasets = [ dataname for dataname in systems[ type ] if datasets_openclosed[ dataname ] == 'open' ] #consider only open datasets

		for pos, dataname in enumerate( sel_datasets ): #loop through (open) datasets
			print( 'dataset name: ' + dataname ) #print dataset		print( 'dataset name: ' + dataname ) #print dataset

			#load parameters and fitted model parameters for samples of original dataset
			params = params_data.loc[ dataname ]
			params_sample = pd.read_pickle( saveloc_samp+'params_sample_'+dataname+'.pkl' )
			params_model = pd.read_pickle( saveloc_samp+'params_model_'+dataname+'.pkl' )
			params_all = pd.concat( [ params_sample, params_model ], axis=1 ) #join

			#filter data (with sampling fraction larger than threshold!)
			params_plot = params_all[ :kmax ]

			#plot plot!
			yplot = params_plot.pnu / ( params_plot.index * intervals_data.loc[dataname, 'tdays'] )
			plt.semilogy( params_plot.index, yplot, c=datacols[ type ], lw=plot_props['linewidth'] )

	#finalise subplot
	plt.axis([ 1, kmax, 1e-6, 2e-2 ])
	ax.tick_params( axis='x', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=3 )
	ax.tick_params( axis='y', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=2 )
	plt.minorticks_off()


# B2: Effect of subsampling in open systems

	print( 'EFFECT OF SUBSAMPLING 2' ) #to know where we stand

	samp_thres = 0.1 #sampling threshold

	#datasets
	sel_datasets = [ 'github-watch-weekly', 'Citations' ]

	formats = {
	'github-watch-weekly' : ( 0.2, 0.5 ),
	'Citations' : ( 0.85, 0.4 ) }

	#initialise subplot
	ax = plt.subplot( subgrid[ 3, : ] )
	sns.despine( ax=ax ) #take out top and right spines
	plt.xlabel( r'$\nu_r$', size=plot_props['xylabel'], labelpad=-3 )
	plt.ylabel( r'$\tau_r$', size=plot_props['xylabel'], labelpad=0 )

	for grid_pos, dataname in enumerate( sel_datasets ): #loop through datasets
		print( 'dataset name: ' + dataname ) #print dataset

		#load parameters and fitted model parameters for samples of original dataset
		params = params_data.loc[ dataname ]
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
		params_plot = params_all[ params_all.samp_frac > samp_thres ]

		#plot plot!

		#(filtered) sampled dataset
		scat = plt.scatter( params_plot['pnu_resc'], params_plot['ptau_resc'], c=params_plot['samp_frac'], vmin=0, vmax=1, cmap='winter_r', s=20, zorder=1 )

		#texts
		plt.text( *formats[dataname], datasets[dataname], va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel']-1, color='0.4' )

	#universal curve in model
	pnu_resc_vals = np.logspace( -3, np.log10( 4e-1 ), 50 ) #pick rescaled pnu as variable
	ptau_resc_vals = 1 / pnu_resc_vals #and slide over universal curve
	plt.loglog( pnu_resc_vals, ptau_resc_vals, '--', c='0.5', lw=plot_props['linewidth'], label=r'$\tau_r \nu_r = 1$', zorder=0 )

	#finalise subplot
	plt.axis([ 1e-3, 4e-1, 2.5e-0, 1e3 ])
	ax.tick_params( axis='x', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=3 )
	ax.tick_params( axis='y', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=2 )
	plt.minorticks_off()

	#colorbar
	cax = plt.subplot( subgrid[ 2, 2: ] )
	cbar = plt.colorbar( scat, cax=cax, orientation='horizontal' )
	plt.text( -0.3, 0, '$T_{\mathrm{eff}} / T=$', va='center', ha='center', transform=cax.transAxes, fontsize=11 )


# C-E: Regimes of Levy/Diffusion/replacement dynamics in open systems

	print( 'REGIMES OF DYNAMICS' ) #to know where we stand

	#datasets
	sel_datasets = [ 'github-watch-weekly', 'english', 'Citations', 'simulated system' ]
	#params for simulation of replacement regime
	params_sim = { 'ptau' : 0.1, 'pnu' : 0.8, 'p0' : 0.1 }
	#plot variables
	colors = sns.color_palette( 'Paired', n_colors=3 )

	#initialise subplot
	subgrid = grid[ 1, : ].subgridspec( 1, 4, wspace=0.2 )

	for grid_pos, dataname in enumerate( sel_datasets ): #loop through datasets
		print( 'dataset name: ' + dataname ) #print dataset

		#initialise subplot
		ax = plt.subplot( subgrid[ grid_pos ] )
		sns.despine( ax=ax ) #take out top and right spines
		plt.xlabel( r'$\nu_r$', size=plot_props['xylabel'], labelpad=-3 )
		if grid_pos == 0:
			plt.ylabel( r'$W_{\bullet}$', size=plot_props['xylabel'], labelpad=2 )

		#subfigure labels
		if grid_pos == 0:
			plt.text( -0.12, 0.95, 'c', va='bottom', ha='right', transform=ax.transAxes, fontsize=plot_props['figlabel'], weight='bold' )
		if grid_pos == 1:
			plt.text( -0.03, 0.95, 'd', va='bottom', ha='right', transform=ax.transAxes, fontsize=plot_props['figlabel'], weight='bold' )
		if grid_pos == 3:
			plt.text( -0.03, 0.95, 'e', va='bottom', ha='right', transform=ax.transAxes, fontsize=plot_props['figlabel'], weight='bold' )

		#prepare data

		#get parameters
		if grid_pos < 3: #real datasets

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

		else: #simulated data

			#simulation parameters
			ptau, pnu, p0 = params_sim['ptau'], params_sim['pnu'], params_sim['p0']
			#get openness derivative from equations
			open_deriv = pnu * ( pnu + ptau ) / ( pnu + p0 * ptau )


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
		handle_param = plt.axvline( pnu_resc, ls='--', c='0.5', label=None, lw=plot_props['linewidth'], zorder=0 )

		#texts
		if grid_pos < 3:
			dset_str = datasets_oneliners[ dataname ]
		else:
			dset_str = dataname
		plt.text( 0.5, 1.1, dset_str, va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		#regime arrows

		if grid_pos == 0:
			txt_str = 'Lévy walk regime\n'+r'$W_{\mathrm{levy}} \gg 0$'
			plt.text( 0.5, 1.6, txt_str, va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['xylabel'], weight='bold', zorder=1 )
			plt.annotate( text='', xy=( 0.5, 1.9 ), xytext=( 0.5, 4 ), arrowprops=dict( headlength=12, headwidth=10, width=5, color=colors[0], alpha=0.5 ), xycoords=ax.transAxes, textcoords=ax.transAxes, zorder=0 )

		if grid_pos == 1:
			txt_str = 'diffusion regime\n'+r'$W_{\mathrm{diff}} \gg 0$'
			plt.text( 1.1, 1.6, txt_str, va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['xylabel'], weight='bold', zorder=1 )
			plt.annotate( text='', xy=( 1.1, 1.9 ), xytext=( 0.5, 2.5 ), arrowprops=dict( headlength=12, headwidth=10, width=5, color=colors[1], alpha=0.5 ), xycoords=ax.transAxes, textcoords=ax.transAxes, zorder=0 )

		if grid_pos == 3:
			txt_str = 'replacement\nregime\n'+r'$W_{\mathrm{repl}} \gg 0$'
			plt.text( 0, 1.6, txt_str, va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['xylabel'], weight='bold', zorder=1 )
			plt.annotate( text='', xy=( -0.25, 1.9 ), xytext=( -0.5, 2.2 ), arrowprops=dict( headlength=12, headwidth=10, width=5, color=colors[2], alpha=0.5 ), xycoords=ax.transAxes, textcoords=ax.transAxes, zorder=0 )

		#legends

		if grid_pos == 0:
			leg1 = plt.legend( loc='center right', bbox_to_anchor=(0.7, 0.5), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'] )
			ax.add_artist(leg1)

			leg2 = plt.legend( ( handle_data, handle_model, handle_param ), ( 'data', 'model', r'$\nu_r$' ), loc='center right', bbox_to_anchor=(1.1, 0.5), prop=plot_props['legend_prop'], handlelength=1.7, numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'] )
			ax.add_artist(leg2)

		if grid_pos == 3:
			plot_str = r'$\tau =$ '+'{}'.format( ptau )+'\n'+r'$\nu =$ '+'{}'.format( pnu )+'\n'+r'$p =$ '+'{}'.format( p0 )
			plt.text( 0.1, 0.5, plot_str, va='center', ha='left', transform=ax.transAxes, fontsize=plot_props['legend_prop']['size'])

		#finalise subplot
		plt.axis([ 1e-3, 1e-0, 0, 1 ])
		ax.tick_params( axis='x', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=3 )
		ax.tick_params( axis='y', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=2 )
		if grid_pos:
			plt.yticks([])
		ax.locator_params( axis='y', nbins=3 ) #change number of ticks in axes
		plt.minorticks_off()


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )

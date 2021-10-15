#! /usr/bin/env python

### SCRIPT FOR PLOTTING SUPP FIGURE (DEVIATIONS) IN FARRANKS PROJECT ###

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
#	datasets = { 'VideogameEarnings' : 'videogame\nplayers', 'Virus' : 'viruses' } #shady data

	datasets_openclosed = { 'AcademicRanking' : 'open', 'AtlasComplex' : 'open', 'Citations' : 'open', 'Cities_RU' : 'open', 'Cities_UK' : 'closed', 'Earthquakes_avgMagnitude' : 'closed', 'Earthquakes_numberQuakes' : 'closed', 'english' : 'open', 'enron-sent-mails-weekly' : 'open', 'FIDEFemale' : 'open', 'FIDEMale' : 'open', 'Football_FIFA' : 'closed', 'Football_Scorers' : 'open', 'Fortune' : 'open', 'french' : 'open', 'german' : 'open', 'github-watch-weekly' : 'open', 'Golf_OWGR' : 'open', 'Hienas' : 'open', 'italian' : 'open', 'metroMex' : 'closed', 'Nascar_BuschGrandNational' : 'open', 'Nascar_WinstonCupGrandNational' : 'open', 'Poker_GPI' : 'open', 'russian' : 'open', 'spanish' : 'open','Tennis_ATP' : 'open', 'TheGuardian_avgRecommends' : 'open', 'TheGuardian_numberComments' : 'open', 'UndergroundByWeek' : 'closed' } #type dict
#	datasets = { 'VideogameEarnings' : 'open', 'Virus' : 'open' } #shady data

	datatypes = { 'AcademicRanking' : 'society', 'AtlasComplex' : 'economics', 'Citations' : 'society', 'Cities_RU' : 'infrastructure', 'Cities_UK' : 'infrastructure', 'Earthquakes_avgMagnitude' : 'nature', 'Earthquakes_numberQuakes' : 'nature', 'english' : 'languages', 'enron-sent-mails-weekly' : 'society', 'FIDEFemale' : 'sports', 'FIDEMale' : 'sports', 'Football_FIFA' : 'sports', 'Football_Scorers' : 'sports', 'Fortune' : 'economics', 'french' : 'languages', 'german' : 'languages', 'github-watch-weekly' : 'society', 'Golf_OWGR' : 'sports', 'Hienas' : 'nature', 'italian' : 'languages', 'metroMex' : 'infrastructure', 'Nascar_BuschGrandNational' : 'sports', 'Nascar_WinstonCupGrandNational' : 'sports', 'Poker_GPI' : 'sports', 'russian' : 'languages', 'spanish' : 'languages','Tennis_ATP' : 'sports', 'TheGuardian_avgRecommends' : 'society', 'TheGuardian_numberComments' : 'society', 'UndergroundByWeek' : 'infrastructure' } #type dict
#	datasets = { 'VideogameEarnings' : 'economics', 'Virus' : 'nature' } #shady data

	palette = sns.color_palette( 'Set2', n_colors=7 ) #selected colormap for types
	datacols = { 'society' : palette[0], 'languages' : palette[1], 'economics' : palette[2], 'infrastructure' : palette[3], 'nature' : palette[4], 'sports' : palette[6] } #set color for dataset type

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 13,
	'text_size' : 10,
	'marker_size' : 6,
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
	'aspect_ratio' : (4, 6),
	'grid_params' : dict( left=0.09, bottom=0.08, right=0.995, top=0.93, wspace=0.3, hspace=0.6 ),
	'dpi' : 300,
	'savename' : 'figure_supp_deviations' }


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


# A: Fractional variations in fitted parameters of bootstrapped model

	# sel_datasets = [ dataname for dataname in fluxmean_data.index if datasets_openclosed[ dataname ] == 'open' ]
	sel_datasets = [ 'AcademicRanking', 'AtlasComplex', 'Citations', 'Cities_RU', 'FIDEFemale', 'FIDEMale', 'Football_Scorers', 'Fortune', 'Golf_OWGR', 'Hienas', 'Nascar_BuschGrandNational', 'Nascar_WinstonCupGrandNational', 'Poker_GPI', 'Tennis_ATP', 'TheGuardian_numberComments', 'enron-sent-mails-weekly' ]

	for grid_pos, dataname in enumerate( sel_datasets ): #loop through (open!) datasets (in order by decreasing mean flux)
		print( 'flux = {:.2f}, dataset = {}'.format( fluxmean_data[ dataname ], dataname ) ) #to know where we stand

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos ] )
		sns.despine( ax=ax ) #take out spines

		#prepare data

		#load fitted model parameters
		params_model = pd.read_pickle( saveloc_data+'params_model_'+dataname+'.pkl' )
		pnu = params_model.loc['optimal', 'pnu']
		ptau = params_model.loc['optimal', 'ptau']
		p0 = params_model.loc['optimal', 'p0']
		open_deriv = params_model.loc['optimal', 'open_deriv']

		#load parameters of bootstrapped samples of model
		params_devs = pd.read_pickle( saveloc_data+'params_devs_'+dataname+'.pkl' )
		#NOTE: for incomplete sampling, just keep finished realisations
		params_devs = params_devs[ params_devs.p0 > 0 ]
		print( '\trealisations: {}'.format( params_devs.index.size ) )

		#statistical significance of deviations

		#get rescaled model parameters (in data)
		pnu_resc = ( pnu - p0 * open_deriv ) / open_deriv
		ptau_resc = ptau / ( p0 * (1 - p0) * open_deriv )

		#get rescaled model parameters (in bootstrapped samples of model)
		pnu_resc_devs = ( ( params_devs.pnu - params_devs.p0 * params_devs.open_deriv ) / params_devs.open_deriv ).rename('pnu_resc')
		ptau_resc_devs = ( params_devs.ptau / ( params_devs.p0 * (1 - params_devs.p0) * params_devs.open_deriv ) ).rename('ptau_resc')

		#get distance from universal curve (in data/samples)
		curve = np.abs( np.log( ptau_resc * pnu_resc ) )
		curve_devs = np.abs( np.log( ptau_resc_devs * pnu_resc_devs ) )

		#p-value as fraction of bootstrapped samples with larger distance than data's
		pvalue = curve_devs[ curve_devs > curve ].size / float( curve_devs.size )
		perror = 1 / ( 2 * np.sqrt( curve_devs.size ) )
		print('\tp-value: {:.2f} +- {:.2f}'.format(pvalue, perror))

		#plot plot!

		#fitted parameters for data
		plt.axhline( ls='--', c='0.5', lw=plot_props['linewidth'], zorder=1 )
		plt.axvline( ls='--', c='0.5', lw=plot_props['linewidth'], zorder=1 )

		#KDE/mean of bootstrapped model

		xplot = ( ( params_devs['pnu'] - pnu ) / pnu ).rename('pnu_frac')
		yplot = ( ( params_devs['ptau'] - ptau ) / ptau ).rename('ptau_frac')
		xyplot = pd.concat( [ xplot, yplot ], axis=1 )

		plt.plot( xplot.mean(), yplot.mean(), 'x', c='r', ms=plot_props['marker_size'], zorder=1 )
		if dataname != 'spanish':
			sns.kdeplot( data=xyplot, x='pnu_frac', y='ptau_frac', ax=ax, fill=True, palette='GnBu', levels=np.linspace(0.1, 1., 10), zorder=0 )

		#texts
		plt.text( 0.5, 1.25, datasets[ dataname ], va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		pvalue_str = '$\Lambda =$ {:.2f} $\pm$ {:.2f}'.format( pvalue, perror )
		plt.text( 0.02, 0.99, pvalue_str, va='top', ha='left', transform=ax.transAxes, fontsize=plot_props['text_size'])

		if grid_pos == 0:
			plt.text( -0.68, 0.5, 'open', va='center', ha='center', transform=ax.transAxes, weight='bold', rotation='vertical', fontsize=plot_props['xylabel'] )
			plt.annotate( text='', xy=( -0.68, -4.5 ), xytext=( -0.68, 0 ), arrowprops=dict(arrowstyle='<-', color='0.6'), xycoords=ax.transAxes, textcoords=ax.transAxes )

		#finalise subplot
		if grid_pos in [ 18, 19, 20, 21, 22, 23 ]:
			plt.xlabel( r'$\Delta \nu$', size=plot_props['xylabel'], labelpad=2 )
		else:
			plt.xlabel('')
		if grid_pos in [ 0, 6, 12, 18 ]:
			plt.ylabel( r'$\Delta \tau$', size=plot_props['xylabel'], labelpad=2 )
		else:
			plt.ylabel('')
		plt.axis([ -0.5, 0.6, -0.7, 0.5 ])
		ax.locator_params( axis='both', nbins=3 )
		ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )
		if grid_pos not in [ 18, 19, 20, 21, 22, 23 ]:
			plt.xticks([])
		if grid_pos not in [ 0, 6, 12, 18 ]:
			plt.yticks([])

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )


#DEBUGGIN'

	#universal curve in model

	# pnu_resc_vals = np.logspace( -3, np.log10( 4e-1 ), 50 ) #pick rescaled pnu as variable
	# ptau_resc_vals = 1 / pnu_resc_vals #and slide over universal curve

		# params_resc_devs = pd.concat( [ pnu_resc_devs, ptau_resc_devs ], axis=1 ) #and join

		#rescaled fitted parameters for data
		# plt.loglog( pnu_resc, ptau_resc, 'o', label='data', c=datacols[datatypes[dataname]], ms=plot_props['marker_size'], zorder=2 )

		# sns.kdeplot( data=params_resc_devs, x='pnu_resc', y='ptau_resc', ax=ax, log_scale=True, fill=True, palette='GnBu', zorder=1 )

		#universal curve in model
		# plt.loglog( pnu_resc_vals, ptau_resc_vals, '--', c='0.5', lw=plot_props['linewidth'], label=r'$\tau_r \nu_r = 1$', zorder=0 )

			# plt.xlabel( r'$\nu_r$', size=plot_props['xylabel'], labelpad=2 )
			# plt.ylabel( r'$\tau_r$', size=plot_props['xylabel'], labelpad=2 )
		# curve = np.abs( ptau_resc - 1/pnu_resc )
		# curve_devs = np.abs( ptau_resc_devs - 1/pnu_resc_devs )
		# curve = np.abs( pnu_resc - 1/ptau_resc )
		# curve_devs = np.abs( pnu_resc_devs - 1/ptau_resc_devs )
		# curve = ( ptau_resc - 1/pnu_resc )**2 + ( pnu_resc - 1/ptau_resc )**2
		# curve_devs = ( ptau_resc_devs - 1/pnu_resc_devs )**2 + ( pnu_resc_devs - 1/ptau_resc_devs )**2

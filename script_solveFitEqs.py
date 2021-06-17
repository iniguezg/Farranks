#! /usr/bin/env python

### SCRIPT FOR SOLVING FIT EQUATIONS IN FARRANKS PROJECT ###

#import modules
import numpy as np
import pandas as pd
import mpmath as mpm
import seaborn as sns
import scipy.optimize as spo
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser


## RUNNING DATA SCRIPT ##

if __name__ == "__main__":

	## CONF ##
	prec = 3 #float precision for printing
	pnu_min, pnu_max, pnu_num = -3, 0, 5000 #parameter array
	tol = 1e-6 #tolerance for plotting

	#flags and locations
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_data = root_loc+'nullModel/v4/files/' #location of output files

	#datasets to explore
#	datasets = [ 'AcademicRanking', 'AtlasComplex', 'Citations', 'Cities_RU', 'Cities_UK', 'Earthquakes_avgMagnitude', 'Earthquakes_numberQuakes', 'english', 'enron-sent-mails-weekly', 'FIDEFemale', 'FIDEMale', 'Football_FIFA', 'Football_Scorers', 'Fortune', 'french', 'german', 'github-watch-weekly', 'Golf_OWGR', 'Hienas', 'italian', 'metroMex', 'Nascar_BuschGrandNational', 'Nascar_WinstonCupGrandNational', 'Poker_GPI', 'russian', 'spanish', 'Tennis_ATP', 'TheGuardian_avgRecommends', 'TheGuardian_numberComments', 'UndergroundByWeek' ]
	datasets = [ 'AcademicRanking', 'AtlasComplex', 'Citations', 'Cities_RU', 'english', 'enron-sent-mails-weekly', 'FIDEFemale', 'FIDEMale', 'Football_FIFA', 'Football_Scorers', 'Fortune', 'french', 'german', 'github-watch-weekly', 'Golf_OWGR', 'Hienas', 'italian', 'Nascar_BuschGrandNational', 'Nascar_WinstonCupGrandNational', 'Poker_GPI', 'russian', 'spanish', 'Tennis_ATP', 'TheGuardian_avgRecommends', 'TheGuardian_numberComments' ]
#	datasets = [ 'AcademicRanking' ]

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 13,
	'text_size' : 10,
	'marker_size' : 2,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':11 },
	'legend_hlen' : 1.8,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	colors = sns.color_palette( 'GnBu', n_colors=3 ) #colors to plot


	for dataname in datasets: #loop through considered datasets
		print( 'dataset name: ' + dataname ) #print dataset

		#plot variables
		fig_props = { 'fig_num' : 1,
		'fig_size' : (10, 5),
		'aspect_ratio' : (1, 2),
		'grid_params' : dict( left=0.075, bottom=0.13, right=0.985, top=0.97, wspace=0.3 ),
		'dpi' : 300,
		'savename' : 'figure_supp_fitEqs_'+dataname }


		## DATA ##

		#get parameters for all datasets and selected dataset
		params_data = pd.read_pickle( saveloc_data+'params_data.pkl' )
		params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		p0 = mpm.mpf(str( N0 )) / mpm.mpf(str( N )) #ranking fraction

		param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

		#mean flux over time/ranks
		fluOprops_data = pd.read_pickle( saveloc_data + 'fluOprops_' + param_str_data )
		#average OUT-/IN- flux over time, and then over ranks
		flux_data = mpm.mpf(str( fluOprops_data.mean( axis=0 ).mean() ))

		#average openness derivative
		openprops_data = pd.read_pickle( saveloc_data + 'openprops_' + param_str_data )
		open_deriv_data = mpm.mpf(str( openprops_data.loc[ 'open_deriv' ].mean() )) #get mean


		## MODEL ##

		pnu_vals = [ mpm.mpf(str( pnu )) for pnu in np.logspace( pnu_min, pnu_max, num=pnu_num ) ] #replacement probability
		pnu_vals_bound = mpm.linspace( p0*open_deriv_data + tol, open_deriv_data - tol, pnu_num )

		#functions for pnu trascendental equation
		exp_func = lambda pnu : mpm.exp( pnu * ( pnu - open_deriv_data ) / ( pnu - p0 * open_deriv_data ) )
		pnu_func = lambda pnu : mpm.log( ( p0 + (1 - p0) * exp_func( pnu ) ) / ( 1 - flux_data ) )

		#ptau equation
		ptau_func = lambda pnu : pnu * ( pnu - open_deriv_data ) / ( p0 * open_deriv_data - pnu )

		#find pnu root
		pnu_lambda = lambda pnu : float( mpm.fabs( pnu - pnu_func( pnu ) ) )
		pnu_res = spo.minimize_scalar( pnu_lambda, bounds=( float( p0*open_deriv_data ), float( open_deriv_data ) ), method='bounded' )
		pnu_star = pnu_res.x

		#find ptau root
		ptau_star = ptau_func( pnu_star )

		#exception: numerical errors
		if ptau_star > 1:
			ptau_star = 1.


		## PRINTING ##

		#equations for flux/openness
		flux_theo = lambda pnu, ptau : 1 - mpm.exp( -pnu ) * ( p0 + (1 - p0) * mpm.exp( -ptau ) )
		open_deriv_theo = lambda pnu, ptau : pnu * ( pnu + ptau ) / ( pnu + p0 * ptau )

		print( 'Mean flux ->\t\tData: '+mpm.nstr( flux_data, 3 )+' \t Equations: '+mpm.nstr( flux_theo( pnu_star, ptau_star ), 3 ) )
		print( 'Avg openness deriv -> Data: '+mpm.nstr( open_deriv_data, 3 )+' \t Equations: '+mpm.nstr( open_deriv_theo( pnu_star, ptau_star ), 3 ) )


		## PLOTTING ##

# A: trascendental equation for pnu

		#initialise plot
		sns.set( style="white" ) #set fancy fancy plot
		fig = plt.figure( fig_props['fig_num'], figsize=np.array( fig_props['fig_size'] ) )
		plt.clf()
		grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
		grid.update( **fig_props['grid_params'] )

		#initialise subplot
		ax = plt.subplot( grid[ 0 ] )
		sns.despine( ax=ax ) #take out top and right spines
		plt.xlabel( r'$\nu$', size=plot_props['xylabel'] )
#		plt.ylabel( r'$\nu$', size=plot_props['xylabel'] )

		#subplot plot!

		yplot = [ pnu_func( pnu ) for pnu in pnu_vals ]
		plt.loglog( pnu_vals, yplot, 'o', c=colors[2], ms=plot_props['marker_size'], label=r'rhs Eq. (S45b)', zorder=2 )

		plt.loglog( pnu_vals, pnu_vals, '--', c=colors[2], lw=plot_props['linewidth'], label=r'lhs Eq. (S45b)', zorder=2 )

#		label = r'$\nu =$ '+mpm.nstr( pnu_star, prec )
		label = r'$\nu = \nu^*$'
		plt.axvline( x=pnu_star, ls='-.', c='0.5', lw=plot_props['linewidth']-1, label=label, zorder=1 )

		plt.axvspan( p0*open_deriv_data, open_deriv_data, facecolor='0.9', lw=plot_props['linewidth']-1, label=r'$p \dot{o} < \nu < \dot{o}$', zorder=0 )

#		#texts
#		plt.text( 1.2, 1.11, dataname, va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )
#		plt.text( 1.2, 1.06, '$N = $ {}, $N_0 = $ {}, $T = $ {}'.format( N, N0, T ), va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#finalise subplot
		plt.axis([ float(min(pnu_vals)), float(max(pnu_vals)), float(min(pnu_vals)), float(max(pnu_vals)) ])
		ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )

		#legend
		plt.legend( loc='lower right', prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'] )


# B: equation for ptau

		#initialise subplot
		ax = plt.subplot( grid[ 1 ] )
		sns.despine( ax=ax ) #take out top and right spines
		plt.xlabel( r'$\nu_r$', size=plot_props['xylabel'] )
		plt.ylabel( r'$\tau_r$', size=plot_props['xylabel'] )
#		plt.xlabel( r'$( \nu - p \langle \dot{o} \rangle ) / \langle \dot{o} \rangle$', size=plot_props['xylabel'] )
#		plt.ylabel( r'$\tau / ( p (1 - p) \langle \dot{o} \rangle )$', size=plot_props['xylabel'] )

		#subplot plot!

		xplot = [ ( pnu - p0 * open_deriv_data ) / open_deriv_data for pnu in pnu_vals_bound ]
		yplot = [ ptau_func( pnu ) / ( p0 * (1 - p0) * open_deriv_data ) for pnu in pnu_vals_bound ]
		plt.loglog( xplot, yplot, 'o', c=colors[2], ms=plot_props['marker_size'], label=r'Eq. (S45a)', zorder=0 )

#		label = r'$\tau =$ '+mpm.nstr( ptau_star, prec )
		label = r'$\tau = \tau^*$'
		plt.axhline( y = ptau_star / ( p0 * (1 - p0) * open_deriv_data ), ls='-.', c='0.5', lw=plot_props['linewidth']-1, label=label, zorder=1 )

#		label = r'$\nu =$ '+mpm.nstr( pnu_star, prec )
		label = r'$\nu = \nu^*$'
		plt.axvline( x = ( pnu_star - p0 * open_deriv_data ) / open_deriv_data, ls='-.', c='0.5', lw=plot_props['linewidth']-1, label=label, zorder=1 )

		xplot = [ ( pnu - p0 * open_deriv_data ) / open_deriv_data for pnu in pnu_vals_bound ]
		yplot = [ open_deriv_data / ( pnu - p0 * open_deriv_data ) for pnu in pnu_vals_bound ]
		plt.plot( xplot, yplot, '--', c='0.5', lw=plot_props['linewidth'], label=r'$\tau_r \nu_r = 1$', zorder=1 )

		#finalise subplot
		plt.axis([ 1e-3, 1e-0, 1e-0, 1e3 ])
		ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )

		#legend
		plt.legend( loc='lower left', prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'] )


		#finalise plot
		if fig_props['savename'] != '':
			plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )

#DEBUGGIN'

#		xplot = pnu_vals_bound
#		plt.axvspan( p0*open_deriv_data, open_deriv_data, facecolor='0.9', lw=plot_props['linewidth']-1, label=r'$p \langle \dot{o} \rangle < \nu < \langle \dot{o} \rangle$', zorder=0 )
#		plt.axis([ float(0.9 * p0 * open_deriv_data), float(1.1 * open_deriv_data), float(min(pnu_vals)), float(max(pnu_vals)) ])
#		plt.axis([ 1e-3, float( 1 - p0 ), 1e-3, float( 1 / ( p0 * (1 - p0) * open_deriv_data ) ) ])

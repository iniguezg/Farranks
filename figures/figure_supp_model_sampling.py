#! /usr/bin/env python

### SCRIPT FOR PLOTTING SUPP FIGURE (MODEL SAMPLING) IN FARRANKS PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import model_misc

## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#model variables

	# ptau_vals = [ 0.1 ]
	# ptau_sel = 0.1
	# pnu_vals = [ 0.03 ]
	# pnu_sel = 0.03
	# N = 2000
	# N0 = 500
	# T = 50

	# #companies
	# ptau_sel = 0.0689
	# pnu_sel = 0.0260
	# ptau_vals, pnu_vals = [ptau_sel], [pnu_sel]
	# N, N0, T = 1895, 500, 51
	# # football scorers
	# ptau_sel = 0.1377
	# pnu_sel = 0.0305
	# ptau_vals, pnu_vals = [ptau_sel], [pnu_sel]
	# N, N0, T = 2397, 400, 53
	# #enron-sent-mails-weekly
	# ptau_sel = 0.7747
	# pnu_sel = 0.0129
	# ptau_vals, pnu_vals = [ptau_sel], [pnu_sel]
	# N, N0, T = 4720, 209, 101
	# #Poker_GPI
	# ptau_sel = 0.0402
	# pnu_sel = 0.0058
	# ptau_vals, pnu_vals = [ptau_sel], [pnu_sel]
	# N, N0, T = 9799, 1795, 221
	#Golf_OWGR
	ptau_sel = 0.0068
	pnu_sel = 0.0012
	ptau_vals, pnu_vals = [ptau_sel], [pnu_sel]
	N, N0, T = 3632, 1150, 768
	# #Tennis_ATP
	# ptau_sel = 0.0137
	# pnu_sel = 0.0021
	# ptau_vals, pnu_vals = [ptau_sel], [pnu_sel]
	# N, N0, T = 4793, 1600, 400
	# #Nascar_BuschGrandNational
	# ptau_sel = 0.4744
	# pnu_sel = 0.0455
	# ptau_vals, pnu_vals = [ptau_sel], [pnu_sel]
	# N, N0, T = 676, 76, 34

	ntimes = 10

	#parameters/properties dicts
	params = { 'N0' : N0, 'T' : T, 'ntimes' : ntimes }
	prop_names = ( 'samp' ) #only compute sampling

	#flag and locations
	loadflag = 'y'
	saveflag = 'y'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_model = root_loc+'nullModel/v4/files/model/sampling/' #location of output files

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 13,
	'text_size' : 8,
	'marker_size' : 6,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':10 },
	'legend_hlen' : 3,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (10, 5),
	'aspect_ratio' : (1, 2),
	'grid_params' : dict( left=0.07, bottom=0.11, right=0.99, top=0.98, wspace=0.1, hspace=0.2 ),
	'dpi' : 300,
	'savename' : 'figure_supp_model_sampling' }


	## PLOTTING ##

	#initialise plot
	sns.set( style="white" ) #set fancy fancy plot
	#convert fig size to inches
	fig = plt.figure( fig_props['fig_num'], figsize=np.array( fig_props['fig_size'] ) )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# A: Effective model parameter (ptau) as a function of sampling jump

	#model parameters
	params[ 'pnu' ] = pnu_sel #set replacement probability

	#subplot variables
	colors = sns.color_palette( 'GnBu', n_colors=len(ptau_vals) ) #colors to plot

	#initialise subplot
	ax = plt.subplot( grid[ 0 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$k$', size=plot_props['xylabel'], labelpad=2 )
	plt.ylabel( r'$\tau / \tau^*$', size=plot_props['xylabel'], labelpad=2 )

	#plot plot!

	for pos_ptau, ptau in enumerate( ptau_vals ): #loop through ptau values

		#model parameters
		params[ 'N' ] = N #set (initial) system size
		params[ 'ptau' ] = ptau #set displacement probability
		print( params )

		#estimate system size from rank openness in model (theo) and use as guess for N
		params[ 'N' ] = model_misc.N_est_theo( params )
		print( '\tN_est_theo = {}'.format( params['N'] ) )

		prop_dict = model_misc.model_props( prop_names, params, loadflag=loadflag, saveloc=saveloc_model, saveflag=saveflag ) #run model
		params['N'] = prop_dict['samp'][0].loc[1, 'N'] #update varying parameters
		print( '\t\tN = {}'.format( params['N'] ) )

		#plot plot!

		xplot = prop_dict['samp'][0].index #sampling jumps

		#MODEL
		yplot_model = prop_dict['samp'][0].ptau / params['ptau'] #eff displacement parameter
		label_model = '{}'.format(ptau) if pos_ptau	== 0 else None
		plt.loglog( xplot, yplot_model, 'o', label=label_model, c=colors[pos_ptau], ms=plot_props['marker_size'], lw=plot_props['linewidth'], zorder=2 )

		#THEO
		yplot_theo = ( 1 - ( 1 - params['ptau'] )**xplot ) / params['ptau'] #eff displacement parameter
		label_theo = 'theo' if pos_ptau	== 0 else None
		plt.loglog( xplot, yplot_theo, '--', label=label_theo, c=colors[pos_ptau], ms=plot_props['marker_size'], lw=plot_props['linewidth'], zorder=1 )

		#identity line
		plt.loglog( xplot, xplot, '--', label=r'$\tau = k \tau^*$', c='0.5', ms=plot_props['marker_size'], lw=plot_props['linewidth'], zorder=0 )


	#text
	plt.text( 0.5, 1, r'$\nu^* =$ '+'{}'.format( pnu_sel ), va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

	#legend
	leg = plt.legend( loc='lower right', bbox_to_anchor=(1, 0), title=r'$\tau^* =$', prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=1 )

	#finalise subplot
	# plt.axis([ -0.01, 1.01, -0.01, 1.01 ])
	ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )


# B: Effective model parameter (pnu) as a function of sampling jump

	#model parameters
	params[ 'ptau' ] = ptau_sel #set displacement probability

	#subplot variables
	colors = sns.color_palette( 'GnBu', n_colors=len(pnu_vals) ) #colors to plot

	#initialise subplot
	ax = plt.subplot( grid[ 1 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$k$', size=plot_props['xylabel'], labelpad=2 )
	plt.ylabel( r'$\nu / \nu^*$', size=plot_props['xylabel'], labelpad=2 )

	#plot plot!

	for pos_pnu, pnu in enumerate( pnu_vals ): #loop through pnu values

		#model parameters
		params[ 'N' ] = N #set (initial) system size
		params[ 'pnu' ] = pnu #set replacement probability
		print( params )

		#estimate system size from rank openness in model (theo) and use as guess for N
		params[ 'N' ] = model_misc.N_est_theo( params )
		print( '\tN_est_theo = {}'.format( params['N'] ) )

		prop_dict = model_misc.model_props( prop_names, params, loadflag=loadflag, saveloc=saveloc_model, saveflag=saveflag ) #run model
		params['N'] = prop_dict['samp'][0].loc[1, 'N'] #update varying parameters
		print( '\t\tN = {}'.format( params['N'] ) )

		#plot plot!

		xplot = prop_dict['samp'][0].index #sampling jumps

		#MODEL
		yplot_model = prop_dict['samp'][0].pnu / params['pnu'] #eff replacement parameter
		label_model = '{}'.format(pnu) if pos_pnu == 0 else None
		plt.loglog( xplot, yplot_model, 'o', label=label_model, c=colors[pos_pnu], ms=plot_props['marker_size'], lw=plot_props['linewidth'], zorder=2 )

		#THEO
		yplot_theo = ( 1 - ( 1 - params['pnu'] )**xplot ) / params['pnu'] #eff replacement parameter
		label_theo = 'theo' if pos_pnu == 0 else None
		plt.loglog( xplot, yplot_theo, '--', label=label_theo, c=colors[pos_pnu], ms=plot_props['marker_size'], lw=plot_props['linewidth'], zorder=1 )

		#identity line
		plt.loglog( xplot, xplot, '--', label=r'$\nu = k \nu^*$', c='0.5', ms=plot_props['marker_size'], lw=plot_props['linewidth'], zorder=0 )


	#text
	plt.text( 0.5, 1, r'$\tau^* =$ '+'{}'.format( ptau_sel ), va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

	#legend
	leg = plt.legend( loc='lower right', bbox_to_anchor=(1, 0), title=r'$\nu^* =$', prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=1 )

	#finalise subplot
	# plt.axis([ -0.01, 1.01, -0.01, 1.01 ])
	ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )

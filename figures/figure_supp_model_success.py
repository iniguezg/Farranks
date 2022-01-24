#! /usr/bin/env python

#Farranks - Exploring rank dynamics in complex systems
#Copyright (C) 2022 Gerardo Iñiguez

### SCRIPT FOR PLOTTING SUPP FIGURE (MODEL RANK SUCCESS) IN FARRANKS PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import model_misc

## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#model variables

	p0_vals = [ 0.1, 0.5, 0.9 ]

	ptau_vals = np.linspace( 0.02, 1, num=50 )
	ptau_sel = 0.5

	pnu_vals = np.linspace( 0.02, 1, num=50 )
	pnu_sel = 0.5

	N = 10000
	T = 10
	ntimes = 10

	thres = 0.5 #threshold to calculate success/surprise measure

	dr = 1 / float( N ) #rank increment

	#store parameters in dict
	params = { 'N' : N, 'T' : T, 'ntimes' : ntimes }

	#flag and locations
	loadflag = 'y'
	saveflag = 'y'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_model = root_loc+'nullModel/v4/files/model/success/' #location of output files

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 13,
	'text_size' : 8,
	'marker_size' : 3,
	'linewidth' : 3,
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
	'grid_params' : dict( left=0.065, bottom=0.1, right=0.99, top=0.98, wspace=0.1, hspace=0.2 ),
	'dpi' : 300,
	'savename' : 'figure_supp_model_success' }


	## PLOTTING ##

	#initialise plot
	sns.set( style="white" ) #set fancy fancy plot
	#convert fig size to inches
	fig = plt.figure( fig_props['fig_num'], figsize=np.array( fig_props['fig_size'] ) )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# A: Ranking size dependence of rank success vs. replacement probability in sims/theo

	#model variables
	pnu = pnu_sel #set replacement probability
	params[ 'pnu' ] = pnu

	#subplot variables
	colors = sns.color_palette( 'GnBu', n_colors=len(p0_vals) ) #colors to plot

	#initialise subplot
	ax = plt.subplot( grid[ 0 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$\tau$', size=plot_props['xylabel'], labelpad=2 )
	plt.ylabel( r'$S^{++}$', size=plot_props['xylabel'], labelpad=2 )

	#plot plot!

	for pos_p0, p0 in enumerate( p0_vals ): #loop through p0 values
		print( 'N = {}, p0 = {}'.format( N, p0 ) ) #print state

		N0 = int( p0*N ) #get ranking size
		params[ 'N0' ] = N0 #update parameter dict

		#set label
		label = '{}'.format( p0 )

		#MODEL/THEO

		xplot = ptau_vals #values of time scale
		yplot_model = np.zeros( len( xplot ) ) #initialise model/theo arrays
		yplot_theo = np.zeros( len( xplot ) )

		for pos_ptau, ptau in enumerate( ptau_vals ): #loop through ptau values
			print( 'ptau = {:.2f}'.format( ptau ) ) #print state

			params[ 'ptau' ] = ptau #set ptau in parameter dict

			success_model, not_used = model_misc.model_props( ['succ'], params, loadflag, saveloc_model, saveflag )['succ']
			yplot_model[ pos_ptau ] = success_model.loc[ thres, 1 ] #success (lag=1) for given threshold

			yplot_theo[ pos_ptau ] = model_misc.success_theo( thres, params )[1] #only lag=1

		#plot plot!

		plt.plot( xplot, yplot_model, 'o', label=label, c=colors[pos_p0], ms=plot_props['marker_size'], rasterized=False )

		plt.plot( xplot, yplot_theo, label=None, c=colors[pos_p0], lw=plot_props['linewidth'], rasterized=False )

	#text
	plt.text( 0.5, 1, r'$\nu =$ '+'{}'.format( pnu ), va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

	#legend
	leg = plt.legend( loc='upper right', bbox_to_anchor=(1, 1), title='$p =$', prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=1 )

	#finalise subplot
	plt.axis([ -0.01, 1.01, -0.01, 1.01 ])
	ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )


# B: Ranking size dependence of rank success vs. diffusion probability in sims/theo

	#model variables
	ptau = ptau_sel #set diffusion probability
	params[ 'ptau' ] = ptau

	#subplot variables
	colors = sns.color_palette( 'GnBu', n_colors=len(p0_vals) ) #colors to plot

	#initialise subplot
	ax = plt.subplot( grid[ 1 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$\nu$', size=plot_props['xylabel'], labelpad=2 )

	#plot plot!

	for pos_p0, p0 in enumerate( p0_vals ): #loop through p0 values
		print( 'N = {}, p0 = {}'.format( N, p0 ) ) #print state

		N0 = int( p0*N ) #get ranking size
		params[ 'N0' ] = N0 #update parameter dict

		#set label
		label = '{}'.format( p0 )

		#MODEL/THEO

		xplot = pnu_vals #values of time scale
		yplot_model = np.zeros( len( xplot ) ) #initialise model/theo arrays
		yplot_theo = np.zeros( len( xplot ) )

		for pos_pnu, pnu in enumerate( pnu_vals ): #loop through pnu values
			print( 'pnu = {:.2f}'.format( pnu ) ) #print state

			params[ 'pnu' ] = pnu #set pnu in parameter dict

			success_model, not_used = model_misc.model_props( ['succ'], params, loadflag, saveloc_model, saveflag )['succ']
			yplot_model[ pos_pnu ] = success_model.loc[ thres, 1 ] #success (lag=1) for given threshold

			yplot_theo[ pos_pnu ] = model_misc.success_theo( thres, params )[1] #only lag=1

		#plot plot!

		plt.plot( xplot, yplot_model, 'o', label=label, c=colors[pos_p0], ms=plot_props['marker_size'], rasterized=False )

		plt.plot( xplot, yplot_theo, label=None, c=colors[pos_p0], lw=plot_props['linewidth'], rasterized=False )

	#text
	plt.text( 0.5, 1, r'$\tau =$ '+'{}'.format( ptau ), va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

	#legend
	leg = plt.legend( loc='upper right', bbox_to_anchor=(1, 1), title='$p =$', prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=1 )

	#finalise subplot
	plt.axis([ -0.01, 1.01, -0.01, 1.01 ])
	ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )
	plt.yticks([])


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )

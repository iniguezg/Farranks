#! /usr/bin/env python

### SCRIPT FOR PLOTTING SUPP FIGURE (MODEL RANK DISPLACEMENT) IN FARRANKS PROJECT ###

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

	ptau_vals = [ 0.1, 0.5, 0.9 ]
	ptau_sel = 0.5

	pnu_vals = [ 0.1, 0.5, 0.9 ]
	pnu_sel = 0.5

	N = 100
	T = 10
	p0 = 0.8
	ntimes = 10000

	pos_rank_vals = [ 1, 5 ] #selected rank positions for plotting

	dr = 1 / float( N ) #rank increment
	N0 = int( p0*N ) #get ranking size

	#store parameters in dict
	params = { 'N' : N, 'N0' : N0, 'T' : T, 'ntimes' : ntimes }

	#flag and locations
	loadflag = 'y'
	saveflag = 'y'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_model = root_loc+'nullModel/v4/files/model/displacement/' #location of output files

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
	'fig_size' : (10, 6),
	'aspect_ratio' : (2, 2),
	'grid_params' : dict( left=0.08, bottom=0.09, right=0.985, top=0.98, hspace=0.1, wspace=0.1 ),
	'dpi' : 300,
	'savename' : 'figure_supp_model_displacement' }


	## PLOTTING ##

	#initialise plot
	sns.set( style="white" ) #set fancy fancy plot
	#convert fig size to inches
	fig = plt.figure( fig_props['fig_num'], figsize=np.array( fig_props['fig_size'] ) )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# A (top row): ptau dependence of rank displacement vs. rank in sims/theo

	#model variables
	pnu = pnu_sel #set replacement probability
	params[ 'pnu' ] = pnu

	#subplot variables
	colors = sns.color_palette( 'coolwarm', n_colors=len(ptau_vals) ) #colors to plot

	#loop through rank positions
	for grid_pos, pos_rank in enumerate( pos_rank_vals ):

		#initialise subplot
		ax = plt.subplot( grid[ 0, grid_pos ] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos == 0:
			plt.ylabel( r'$P_{x, t}$', size=plot_props['xylabel'], labelpad=2 )

		#plot plot!

		xplot = np.linspace( 1. / N0, 1, num=N0 ) #values of displaced rank X / N_0

		for pos_ptau, ptau in enumerate( ptau_vals ): #loop through ptau values
			print( 'ptau = {:.2f}'.format( ptau ) ) #print state

			params[ 'ptau' ] = ptau #set ptau in parameter dict

			#set label
			label = '{}'.format( ptau )

			#MODEL/THEO

			dispprops_model, = model_misc.model_props( ['disp'], params, loadflag, saveloc_model, saveflag )['disp']
			yplot_model = dispprops_model.iloc[ :, pos_rank ] #selected r only

			r = dr * ( dispprops_model.columns[ pos_rank ] + 1 ) #get r value
			sel_rank = int( N*r ) - 1 #selected rank (integer)
			yplot_theo = model_misc.displacement_theo( r, params )

			print( 'r = {:.2f}'.format( r ) ) #print state

			#plot plot!

			plt.semilogy( xplot, yplot_model, 'o', label=label, c=colors[ pos_ptau ], ms=plot_props['marker_size'], rasterized=False )

			plt.semilogy( xplot, yplot_theo, label=None, c=colors[ pos_ptau ], lw=plot_props['linewidth'], rasterized=False )

			#lines
			if pos_ptau == 0:
				plt.vlines( r/p0, 0.0002, yplot_theo[sel_rank], ls='--', colors='0.8', label=None, lw=plot_props['linewidth']-1 )

		#text
		plt.text( 0.5, 1, r'$\nu =$ '+'{}'.format( pnu ), va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#legend
		leg = plt.legend( loc='upper right', bbox_to_anchor=(1, 1), title=r'$\tau =$', prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=1 )

		#finalise subplot
		plt.axis([ -0.01, 1.01, 0.0002, 1.2 ])
		ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )
		plt.xticks([])
		if grid_pos == 1:
			plt.yticks([])


# B (bottom row): pnu dependence of rank displacement vs. rank in sims/theo

	#model variables
	ptau = ptau_sel #set diffusion probability
	params[ 'ptau' ] = ptau

	#subplot variables
	colors = sns.color_palette( 'coolwarm', n_colors=len(pnu_vals) ) #colors to plot

	#loop through rank positions
	for grid_pos, pos_rank in enumerate( pos_rank_vals ):

		#initialise subplot
		ax = plt.subplot( grid[ 1, grid_pos ] )
		sns.despine( ax=ax ) #take out spines
		plt.xlabel( r'$x/p$', size=plot_props['xylabel'], labelpad=2 )
		if grid_pos == 0:
			plt.ylabel( r'$P_{x, t}$', size=plot_props['xylabel'], labelpad=2 )

		#plot plot!

		xplot = np.linspace( 1. / N0, 1, num=N0 ) #values of displaced rank X / N_0

		for pos_pnu, pnu in enumerate( pnu_vals ): #loop through pnu values
			print( 'pnu = {:.2f}'.format( pnu ) ) #print state

			params[ 'pnu' ] = pnu #set pnu in parameter dict

			#set label
			label = '{}'.format( pnu )

			#MODEL/THEO

			dispprops_model, = model_misc.model_props( ['disp'], params, loadflag, saveloc_model, saveflag )['disp']
			yplot_model = dispprops_model.iloc[ :, pos_rank ] #selected r only

			r = dr * ( dispprops_model.columns[ pos_rank ] + 1 ) #get r value
			sel_rank = int( N*r ) - 1 #selected rank (integer)
			yplot_theo = model_misc.displacement_theo( r, params )

			print( 'r = {:.2f}'.format( r ) ) #print state

			#plot plot!

			plt.semilogy( xplot, yplot_model, 'o', label=label, c=colors[ pos_pnu ], ms=plot_props['marker_size'], rasterized=False )

			plt.semilogy( xplot, yplot_theo, label=None, c=colors[ pos_pnu ], lw=plot_props['linewidth'], rasterized=False )

			#lines
			if pos_pnu == 0:
				plt.vlines( r/p0, 0.0002, yplot_theo[sel_rank], ls='--', colors='0.8', label=None, lw=plot_props['linewidth']-1 )

		#text
		plt.text( 0.5, 1, r'$\tau =$ '+'{}'.format( ptau ), va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#legend
		leg = plt.legend( loc='upper right', bbox_to_anchor=(1, 1), title=r'$\nu =$', prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=1 )

		#finalise subplot
		plt.axis([ -0.01, 1.01, 0.0002, 1.2 ])
		ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )
		if grid_pos == 1:
			plt.yticks([])


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )

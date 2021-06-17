#! /usr/bin/env python

### SCRIPT FOR PLOTTING SUPP FIGURE (MODEL RANK OPENNESS) IN FARRANKS PROJECT ###

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

	ptau_vals = [ 0.05, 0.1, 0.2 ]
	ptau_sel = 0.1

	pnu_vals = [ 0.05, 0.1, 0.2 ]
	pnu_sel = 0.1

	N = 100
	T = 100
	p0 = 0.1
	ntimes = 100

	dr = 1 / float( N ) #rank increment
	N0 = int( p0*N ) #get ranking size

	#store parameters in dict
	params = { 'N' : N, 'N0' : N0, 'T' : T, 'ntimes' : ntimes }

	#flag and locations
	loadflag = 'y'
	saveflag = 'y'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_model = root_loc+'nullModel/v4/files/model/openness/' #location of output files

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
	'grid_params' : dict( left=0.065, bottom=0.105, right=0.99, top=0.98, wspace=0.1, hspace=0.2 ),
	'dpi' : 300,
	'savename' : 'figure_supp_model_openness' }


	## PLOTTING ##

	#initialise plot
	sns.set( style="white" ) #set fancy fancy plot
	#convert fig size to inches
	fig = plt.figure( fig_props['fig_num'], figsize=np.array( fig_props['fig_size'] ) )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# A: ptau dependence of rank openness vs. time in sims/theo

	#model variables
	pnu = pnu_sel #set replacement probability
	params[ 'pnu' ] = pnu

	#subplot variables
	colors = sns.color_palette( 'GnBu', n_colors=len(ptau_vals) ) #colors to plot

	#initialise subplot
	ax = plt.subplot( grid[ 0 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$t / T$', size=plot_props['xylabel'], labelpad=2 )
	plt.ylabel( r'$o_t$', size=plot_props['xylabel'], labelpad=2 )

	#plot plot!

	xplot = np.linspace( 0, (T-1) / float(T), num=T ) #normalised time = ( 0, ..., T-1 ) / T

	for pos_ptau, ptau in enumerate( ptau_vals ): #loop through ptau values
		print( 'ptau = {:.2f}'.format( ptau ) ) #print state

		params[ 'ptau' ] = ptau #set ptau in parameter dict

		#set label
		label = '{}'.format( ptau )

		#MODEL/THEO

		openprops_model, = model_misc.model_props( ['open'], params, loadflag, saveloc_model, saveflag )['open']
		yplot_model = openprops_model.loc[ 'openness' ]

		yplot_theo = model_misc.openness_theo( params )

		#plot plot!

		plt.semilogy( xplot, yplot_model, 'o', label=label, c=colors[ pos_ptau ], ms=plot_props['marker_size'], rasterized=False )

		plt.semilogy( xplot, yplot_theo, label=None, c=colors[ pos_ptau ], lw=plot_props['linewidth'], rasterized=False )

	#text
	plt.text( 0.5, 0.01, r'$\nu =$ '+'{}'.format( pnu )+'\n'+'($p =$ {})'.format( p0 ), va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

	#legend
	leg = plt.legend( loc='lower right', bbox_to_anchor=(1, 0), title=r'$\tau =$', prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=1 )

	#finalise subplot
	plt.axis([ -0.02, 1.02, 0.8, 30 ])
	ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )


# B: pnu dependence of rank openness vs. time in sims/theo

	#model variables
	ptau = ptau_sel #set diffusion probability
	params[ 'ptau' ] = ptau

	#subplot variables
	colors = sns.color_palette( 'GnBu', n_colors=len(pnu_vals) ) #colors to plot

	#initialise subplot
	ax = plt.subplot( grid[ 1 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$t / T$', size=plot_props['xylabel'], labelpad=2 )

	#plot plot!

	xplot = np.linspace( 0, (T-1) / float(T), num=T ) #normalised time = ( 0, ..., T-1 ) / T

	for pos_pnu, pnu in enumerate( pnu_vals ): #loop through pnu values
		print( 'pnu = {:.2f}'.format( pnu ) ) #print state

		params[ 'pnu' ] = pnu #set pnu in parameter dict

		#set label
		label = '{}'.format( pnu )

		#MODEL/THEO

		openprops_model, = model_misc.model_props( ['open'], params, loadflag, saveloc_model, saveflag )['open']
		yplot_model = openprops_model.loc[ 'openness' ]

		yplot_theo = model_misc.openness_theo( params )

		#plot plot!

		plt.semilogy( xplot, yplot_model, 'o', label=label, c=colors[ pos_pnu ], ms=plot_props['marker_size'], rasterized=False )

		plt.semilogy( xplot, yplot_theo, label=None, c=colors[ pos_pnu ], lw=plot_props['linewidth'], rasterized=False )

	#text
	plt.text( 0.5, 0.01, r'$\tau =$ '+'{}'.format( ptau )+'\n'+'($p =$ {})'.format( p0 ), va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

	#legend
	leg = plt.legend( loc='lower right', bbox_to_anchor=(1, 0), title=r'$\nu =$', prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=1 )

	#finalise subplot
	plt.axis([ -0.02, 1.02, 0.8, 30 ])
	ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )
	plt.yticks([])


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )

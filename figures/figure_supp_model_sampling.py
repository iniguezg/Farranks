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
import model_misc as mm

## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#model parameters
	params = { 'ptau' : 0.01, 'pnu' : 0.01, 'N' : 1000, 'N0' : 500, 'T' : 1000, 'ntimes' : 10 }

	thres = 0.5 #threshold to calculate success/surprise measure
	samp_thres = 0.05 #sampling threshold

	#properties dict
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
	'fig_size' : (10, 6),
	'aspect_ratio' : (2, 3),
	'grid_params' : dict( left=0.065, bottom=0.08, right=0.985, top=0.95, wspace=0.4, hspace=0.2 ),
	'dpi' : 300,
	'savename' : 'figure_supp_model_sampling' }

	colors = sns.color_palette( 'GnBu', n_colors=1 ) #colors to plot


	## MODEL ##

	#get avg samp properties for rank/element time series of null model
	prop_dict = mm.model_props( prop_names, params, loadflag=loadflag, saveloc=saveloc_model, saveflag=saveflag )

	#get sampling jumps and effective times
	k_vals = prop_dict['samp'][0].index.values #sampling jump values
	Teff_vals = np.ceil( params['T'] / k_vals )

	#only consider samples w/ high enough number of observations
	k_vals = k_vals[ Teff_vals / float( params['T'] ) > samp_thres ]
	Teff_vals = np.ceil( params['T'] / k_vals )

	#get effective parameters (due to sampling)
	ptau_eff_vals = 1 - ( 1 - params['ptau'] )**k_vals
	pnu_eff_vals = 1 - ( 1 - params['pnu'] )**k_vals

	#set dict of effective parameters
	params_eff = {
		'ptau' : ptau_eff_vals, 'pnu' : pnu_eff_vals, 'T' : Teff_vals,
		'N' : params['N'], 'N0' : params['N0'],
		'p0' : params['N0'] / float( params['N'] )
	}


	## PLOTTING ##

	#initialise plot
	sns.set( style="white" ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=np.array( fig_props['fig_size'] ) )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# A: Effective mean flux as a function of sampling jump

	#initialise subplot
	ax = plt.subplot( grid[ 0 ] )
	sns.despine( ax=ax ) #take out spines
	plt.ylabel( r'$F_k / F$', size=plot_props['xylabel'], labelpad=2 )

	xplot = k_vals #sampling jumps

	#MODEL
	yplot_model = prop_dict['samp'][0].loc[k_vals, 'flux'] / prop_dict['samp'][0].loc[1, 'flux'] #mean flux
	plt.plot( xplot, yplot_model, 'o', c=colors[0], ms=plot_props['marker_size'], zorder=0 )

	#THEO
	yplot_theo = mm.flux_theo( params_eff ) / mm.flux_theo( params ) #mean flux
	plt.plot( xplot, yplot_theo, '-', c='0.5', lw=plot_props['linewidth'], zorder=1 )

	#finalise subplot
	plt.axis([ xplot[0]-1, xplot[-1]+1, 0, 20 ])
	ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )
	plt.xticks([])


# B: Effective mean openness rate as a function of sampling jump

	#initialise subplot
	ax = plt.subplot( grid[ 1 ] )
	sns.despine( ax=ax ) #take out spines
	plt.ylabel( r'$\dot{o}_k / \dot{o}$', size=plot_props['xylabel'], labelpad=2 )

	xplot = k_vals #sampling jumps

	#MODEL
	yplot_model = prop_dict['samp'][0].loc[k_vals, 'open_deriv'] / prop_dict['samp'][0].loc[1, 'open_deriv'] #mean openness rate
	plt.plot( xplot, yplot_model, 'o', c=colors[0], ms=plot_props['marker_size'], zorder=0 )

	#THEO
	yplot_theo = mm.open_deriv_theo( params_eff ) / mm.open_deriv_theo( params ) #mean openness rate
	plt.plot( xplot, yplot_theo, '-', c='0.5', lw=plot_props['linewidth'], zorder=1 )

	#text
	text_str = r'$\tau =$ '+str(params['ptau'])+r', $\nu =$ '+str(params['pnu'])+' ($p =$ {})'.format( params_eff['p0'] )
	plt.text( 0.5, 1.03, text_str, va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

	#finalise subplot
	plt.axis([ xplot[0]-1, xplot[-1]+1, 0, 20 ])
	ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )
	plt.xticks([])


# C: Effective success as a function of sampling jump

	#initialise subplot
	ax = plt.subplot( grid[ 2 ] )
	sns.despine( ax=ax ) #take out spines
	plt.ylabel( r'$S^{++}_k / S^{++}$', size=plot_props['xylabel'], labelpad=2 )

	xplot = k_vals #sampling jumps

	#MODEL
	yplot_model = prop_dict['samp'][0].loc[k_vals, 'success'] / prop_dict['samp'][0].loc[1, 'success'] #success
	plt.plot( xplot, yplot_model, 'o', c=colors[0], ms=plot_props['marker_size'], zorder=0 )

	#THEO
	yplot_theo = mm.success_time_theo( thres, 1, params_eff ) / mm.success_time_theo( thres, 1, params ) #success
	plt.plot( xplot, yplot_theo, '-', c='0.5', lw=plot_props['linewidth'], zorder=1 )

	#finalise subplot
	plt.axis([ xplot[0]-1, xplot[-1]+1, 0, 1.05 ])
	ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )
	plt.xticks([])


# D: Effective ranking fraction as a function of sampling jump

	#initialise subplot
	ax = plt.subplot( grid[ 3 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$k$', size=plot_props['xylabel'], labelpad=2 )
	plt.ylabel( r'$p_k / p$', size=plot_props['xylabel'], labelpad=2 )

	xplot = k_vals #sampling jumps

	#MODEL
	yplot_model = prop_dict['samp'][0].loc[k_vals, 'p0'] / prop_dict['samp'][0].loc[1, 'p0'] #ranking fraction
	plt.plot( xplot, yplot_model, 'o', c=colors[0], ms=plot_props['marker_size'], zorder=0 )

	#THEO
	yplot_theo = ( 1 / mm.openness_time_theo( params_eff ) ) / ( 1 / mm.openness_time_theo( params ) ) #ranking fraction
	plt.plot( xplot, yplot_theo, '-', c='0.5', lw=plot_props['linewidth'], zorder=1 )

	#finalise subplot
	plt.axis([ xplot[0]-1, xplot[-1]+1, 0, 2 ])
	ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )


# E: Effective displacement parameter as a function of sampling jump

	#initialise subplot
	ax = plt.subplot( grid[ 4 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$k$', size=plot_props['xylabel'], labelpad=2 )
	plt.ylabel( r'$\tau^*_k / \tau^*$', size=plot_props['xylabel'], labelpad=2 )

	xplot = k_vals #sampling jumps

	#MODEL
	yplot_model = prop_dict['samp'][0].loc[k_vals, 'ptau'] / prop_dict['samp'][0].loc[1, 'ptau'] #displacement parameter
	plt.plot( xplot, yplot_model, 'o', c=colors[0], ms=plot_props['marker_size'], zorder=0 )

	#THEO
	yplot_theo = params_eff['ptau'] / params['ptau'] #displacement parameter
	plt.plot( xplot, yplot_theo, '-', c='0.5', lw=plot_props['linewidth'], zorder=1 )

	#finalise subplot
	plt.axis([ xplot[0]-1, xplot[-1]+1, 0, 20 ])
	ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )


 # F: Effective replacement parameter as a function of sampling jump

	#initialise subplot
	ax = plt.subplot( grid[ 5 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$k$', size=plot_props['xylabel'], labelpad=2 )
	plt.ylabel( r'$\nu^*_k / \nu^*$', size=plot_props['xylabel'], labelpad=2 )

	xplot = k_vals #sampling jumps

	#MODEL
	yplot_model = prop_dict['samp'][0].loc[k_vals, 'pnu'] / prop_dict['samp'][0].loc[1, 'pnu'] #displacement parameter
	plt.plot( xplot, yplot_model, 'o', c=colors[0], ms=plot_props['marker_size'], zorder=0 )

	#THEO
	yplot_theo = params_eff['pnu'] / params['pnu'] #displacement parameter
	plt.plot( xplot, yplot_theo, '-', c='0.5', lw=plot_props['linewidth'], zorder=1 )

	#finalise subplot
	plt.axis([ xplot[0]-1, xplot[-1]+1, 0, 20 ])
	ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
#		plt.savefig( fig_props['savename']+'.png', format='png', dpi=fig_props['dpi'] )


#DEBUGGIN'

	# params = { 'ptau' : 0.0068, 'pnu' : 0.0012, 'N' : 2619, 'N0' : 1150, 'T' : 768, 'ntimes' : 10 }
	# params = { 'ptau' : 0.0137, 'pnu' : 0.0021, 'N' : 3438, 'N0' : 1600, 'T' : 400, 'ntimes' : 10 }
	# params = { 'ptau' : 0.05, 'pnu' : 0.05, 'N' : 100, 'N0' : 10, 'T' : 1000, 'ntimes' : 10 }

	# ptau_eff_vals = params['ptau'] * k_vals
	# pnu_eff_vals = params['pnu'] * k_vals
	# ptau_eff_vals = params['N']*( 1 - ( 1 - params['ptau']/params['N'] )**k_vals )
	# pnu_eff_vals = params['N']*( 1 - ( 1 - params['pnu']/params['N'] )**k_vals )
	# ptau_eff_vals[ ptau_eff_vals > 1 ] = 1
	# pnu_eff_vals[ pnu_eff_vals > 1 ] = 1

#	yplot_theo = mm.flux_time_theo( xplot, params ) #mean flux
	# yplot_theo = params_eff['pnu'] * ( params_eff['pnu'] + params_eff['ptau'] ) / ( params_eff['pnu'] + params_eff['p0'] * params_eff['ptau'] )

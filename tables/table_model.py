#! /usr/bin/env python

### SCRIPT FOR CREATING MODEL TABLE IN FARRANKS PROJECT ###

#import modules
import os, sys
import pandas as pd
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc


## RUNNING TABLE SCRIPT ##

if __name__ == "__main__":

	## CONF ##
	prec = 4 #float precision for printing

	thres = 0.5 #threshold to calculate success/surprise measure

	#flag and locations
	loadflag = 'y'
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_data = root_loc+'nullModel/v4/files/' #location of output files

	#datasets to explore
	datasets = [ 'AcademicRanking', 'AtlasComplex', 'Citations', 'Cities_RU', 'Cities_UK', 'Earthquakes_avgMagnitude', 'Earthquakes_numberQuakes', 'english', 'enron-sent-mails-weekly', 'FIDEFemale', 'FIDEMale', 'Football_FIFA', 'Football_Scorers', 'Fortune', 'french', 'german', 'github-watch-weekly', 'Golf_OWGR', 'Hienas', 'italian', 'metroMex', 'Nascar_BuschGrandNational', 'Nascar_WinstonCupGrandNational', 'Poker_GPI', 'russian', 'spanish', 'Tennis_ATP', 'TheGuardian_avgRecommends', 'TheGuardian_numberComments', 'UndergroundByWeek' ]
#	datasets = [ 'VideogameEarnings', 'Virus' ] #shady data

	datatypes = { 'AcademicRanking' : 'open', 'AtlasComplex' : 'open', 'Citations' : 'open', 'Cities_RU' : 'open', 'Cities_UK' : 'closed', 'Earthquakes_avgMagnitude' : 'closed', 'Earthquakes_numberQuakes' : 'closed', 'english' : 'open', 'enron-sent-mails-weekly' : 'open', 'FIDEFemale' : 'open', 'FIDEMale' : 'open', 'Football_FIFA' : 'closed', 'Football_Scorers' : 'open', 'Fortune' : 'open', 'french' : 'open', 'german' : 'open', 'github-watch-weekly' : 'open', 'Golf_OWGR' : 'open', 'Hienas' : 'open', 'italian' : 'open', 'metroMex' : 'closed', 'Nascar_BuschGrandNational' : 'open', 'Nascar_WinstonCupGrandNational' : 'open', 'Poker_GPI' : 'open', 'russian' : 'open', 'spanish' : 'open','Tennis_ATP' : 'open', 'TheGuardian_avgRecommends' : 'open', 'TheGuardian_numberComments' : 'open', 'UndergroundByWeek' : 'closed' } #type dict
#	datasets = { 'VideogameEarnings' : 'open', 'Virus' : 'open' } #shady data


	## DATA ##

	#get parameters for all datasets
	params_data = pd.read_pickle( saveloc_data+'params_data.pkl' )
	intervals_data = pd.read_pickle( saveloc_data+'intervals_data.pkl' )


	params_model_tuples = {} #initialise dict of model params tuples
	for dataname in datasets: #loop through considered datasets

		#prepare data

		#get parameters for dataset
		params = dict( params_data.loc[ dataname ] ) #(dict to have ints and floats!)
		N, N0, T = params['N'], params['N0'], params['T'] #get parameters from data
		param_str_data = dataname+'_N{}_N0{}_T{}.pkl'.format( N, N0, T )

		p0 = N0 / float( N ) #ranking fraction

		#mean flux in data
		fluOprops_data = pd.read_pickle( saveloc_data + 'fluOprops_' + param_str_data )
		flux_data = fluOprops_data.mean( axis=0 ).mean() #mean flux

		#average openness derivative in data
		openprops_data = pd.read_pickle( saveloc_data + 'openprops_' + param_str_data )
		open_deriv_data = openprops_data.loc[ 'open_deriv' ].mean() #get mean

		#success (lag l=1) in data
		success_data = pd.read_pickle( saveloc_data + 'success_' + param_str_data ).loc[ thres ][ 1 ]

		#prepare model

		#model fit parameters
		datatype = datatypes[ dataname ] #dataset type: open, closed

		#get model parameters for selected dataset
		params_model = data_misc.data_estimate_params_all( dataname, params, loadflag, saveloc_data, datatype=datatype )
		params[ 'pnu' ], params[ 'ptau' ] = params_model.loc[ 'optimal', [ 'pnu', 'ptau' ] ] #set parameters

		#rate of element replacement for open systems
		if datatypes[dataname] == 'open':
			rate_repl = params[ 'pnu' ] / intervals_data.loc[dataname, 'tdays']
		else:
			rate_repl = 0.

		#store everything in tuple (in order defined by table!)
		params_model_tuples[ dataname ] = tuple([ flux_data, open_deriv_data, success_data, p0, params[ 'ptau' ], params[ 'pnu' ], rate_repl ])


	## PRINTING ##

	print(
r"""
\begin{table}[!ht]
\small
\noindent\makebox[\textwidth]{
\begin{tabular}{l | r r r | r r r | r }
\toprule
& \multicolumn{3}{c}{Data measure} & \multicolumn{4}{c}{Model parameter} \\
\cmidrule(l){2-8}
Dataset & $F$ & $\dot{o}$ & $S^{++}$ & $p$ & $\tau$ & $\nu$ & \change{$\nu / \ell$ (days$^{-1}$)} \\
\midrule
{\bf Society} & & & & & & & \\
\cmidrule(l){1-1}
GitHub repositories~\cite{github2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['github-watch-weekly'] )+
r"""
The Guardian readers (recc)~\cite{guardian2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['TheGuardian_avgRecommends'] )+
r"""
The Guardian readers (comm)~\cite{guardian2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['TheGuardian_numberComments'] )+
r"""
Enron emails~\cite{enron2015}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['enron-sent-mails-weekly'] )+
r"""
Scientists~\cite{sinatra2015,sinatra2016}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['Citations'] )+
r"""
Universities~\cite{shanghai2016}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['AcademicRanking'] )+
r"""
\midrule
{\bf Languages} & & & & & & & \\
\cmidrule(l){1-1}
Russian~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['russian'] )+
r"""
Spanish~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['spanish'] )+
r"""
German~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['german'] )+
r"""
French~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['french'] )+
r"""
Italian~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['italian'] )+
r"""
English~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['english'] )+
r"""
\midrule
{\bf Economics} & & & & & & & \\
\cmidrule(l){1-1}
Companies~\cite{fortune2005}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['Fortune'] )+
r"""
Countries~\cite{atlas2018,hidalgo2009,hausmann2014}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['AtlasComplex'] )+
r"""
\midrule
{\bf Infrastructure} & & & & & & & \\
\cmidrule(l){1-1}
Cities (RU)~\cite{cottineau2016}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['Cities_RU'] )+
r"""
Metro stations (London)~\cite{murcio2015}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['UndergroundByWeek'] )+
r"""
Cities (GB)~\cite{edwards2016}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['Cities_UK'] )+
r"""
Metro stations (Mexico)"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['metroMex'] )+
r"""
\midrule
{\bf Nature} & & & & & & & \\
\cmidrule(l){1-1}
Hyenas~\cite{ilany2015}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['Hienas'] )+
r"""
Regions JP (quake mag)~\cite{junec2018,karsai2012}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['Earthquakes_avgMagnitude'] )+
r"""
Regions JP (quakes)~\cite{junec2018,karsai2012}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['Earthquakes_numberQuakes'] )+
r"""
\midrule
{\bf Sports} & & & & & & & \\
\cmidrule(l){1-1}
Chess players (male)~\cite{fide2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['FIDEMale'] )+
r"""
Chess players (female)~\cite{fide2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['FIDEFemale'] )+
r"""
Poker players~\cite{poker2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['Poker_GPI'] )+
r"""
Tennis players~\cite{tennis2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['Tennis_ATP'] )+
r"""
Golf players~\cite{golf2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['Golf_OWGR'] )+
r"""
Football scorers~\cite{football2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['Football_Scorers'] )+
r"""
NASCAR drivers (Busch)~\cite{nascar2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['Nascar_BuschGrandNational'] )+
r"""
NASCAR drivers (Winston Cup)~\cite{nascar2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['Nascar_WinstonCupGrandNational'] )+
r"""
National football teams~\cite{teams2018}"""
+r' & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.6f} \\'.format( *params_model_tuples['Football_FIFA'] )+
r"""
\bottomrule
\end{tabular}}
\caption{\small {\bf Data measures and fitted model parameters}. Values of empirical measures used in the fitting process (mean flux $F$, mean \change{turnover} rate $\mod$, and \change{inertia} $S^{++}$) and of fitted model parameters (relative ranking list size $\p0$, displacement probability $\ptau$, and replacement probability $\pnu$) for all considered datasets (see \tref{tab:datasets} and \sref{sec:data}). For open ranking lists, we fit the model to empirical data by setting $N = N_{T-1}$ (i.e. $\p0 = N_0 / N_{T-1}$, see \sref{ssec:modelDef}) and by computing $\ptau$ and $\pnu$ numerically from \eref{eq:paramsSystEqs} in terms of $F$ and $\mod$ (\fref{fig:fitting}). For closed ranking lists, we set $N = N_0$ and $\pnu = 0$, and obtain $\ptau$ explicitly from \eref{eq:fittingClosedSyst} in terms of $S^{++}$ (for time $t=1$ and threshold $c = 0.5$). \change{We also show the rate of element replacement $\nu / \ell$, with $\ell$ the real time between observations (see \tref{tab:datasets})}. Datasets are classified by an (arbitrary) system type based on the nature of the elements in the ranking list.}
\label{tab:parameters}
\end{table}
"""
	)

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

		#store everything in tuple (in order defined by table!)
		params_model_tuples[ dataname ] = tuple([ flux_data, open_deriv_data, success_data, p0, params[ 'ptau' ], params[ 'pnu' ] ])


	## PRINTING ##

	print(
r"""
\begin{table}[!ht]
\small
\noindent\makebox[\textwidth]{
\begin{tabular}{l | r r r | r r r }
\toprule
& \multicolumn{3}{c}{Data measure} & \multicolumn{3}{c}{Model parameter} \\
\cmidrule(l){2-7}
Dataset & $F$ & $\langle \dot{o} \rangle$ & $S^{++}$ & $p_0$ & $p_{\tau}$ & $p_{\nu}$ \\
\midrule
{\bf Society} & & & & & & \\
\cmidrule(l){1-1}
GitHub repositories~\cite{github2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['github-watch-weekly'], prec=prec )+
r"""
The Guardian readers (recc)~\cite{guardian2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['TheGuardian_avgRecommends'], prec=prec )+
r"""
The Guardian readers (comm)~\cite{guardian2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['TheGuardian_numberComments'], prec=prec )+
r"""
Enron emails~\cite{enron2015}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['enron-sent-mails-weekly'], prec=prec )+
r"""
Scientists~\cite{sinatra2015,sinatra2016}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['Citations'], prec=prec )+
r"""
Universities~\cite{shanghai2016}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['AcademicRanking'], prec=prec )+
r"""
\midrule
{\bf Languages} & & & & & & \\
\cmidrule(l){1-1}
Russian~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['russian'], prec=prec )+
r"""
Spanish~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['spanish'], prec=prec )+
r"""
German~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['german'], prec=prec )+
r"""
French~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['french'], prec=prec )+
r"""
Italian~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['italian'], prec=prec )+
r"""
English~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['english'], prec=prec )+
r"""
\midrule
{\bf Economics} & & & & & & \\
\cmidrule(l){1-1}
Companies~\cite{fortune2005}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['Fortune'], prec=prec )+
r"""
Countries~\cite{atlas2018,hidalgo2009,hausmann2014}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['AtlasComplex'], prec=prec )+
r"""
\midrule
{\bf Infrastructure} & & & & & & \\
\cmidrule(l){1-1}
Cities (RU)~\cite{cottineau2016}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['Cities_RU'], prec=prec )+
r"""
Metro stations (London)~\cite{murcio2015}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['UndergroundByWeek'], prec=prec )+
r"""
Cities (GB)~\cite{edwards2016}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['Cities_UK'], prec=prec )+
r"""
Metro stations (Mexico)"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['metroMex'], prec=prec )+
r"""
\midrule
{\bf Nature} & & & & & & \\
\cmidrule(l){1-1}
Hyenas~\cite{ilany2015}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['Hienas'], prec=prec )+
r"""
Regions JP (quake mag)~\cite{junec2018,karsai2012}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['Earthquakes_avgMagnitude'], prec=prec )+
r"""
Regions JP (quakes)~\cite{junec2018,karsai2012}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['Earthquakes_numberQuakes'], prec=prec )+
r"""
\midrule
{\bf Sports} & & & & & & \\
\cmidrule(l){1-1}
Chess players (male)~\cite{fide2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['FIDEMale'], prec=prec )+
r"""
Chess players (female)~\cite{fide2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['FIDEFemale'], prec=prec )+
r"""
Poker players~\cite{poker2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['Poker_GPI'], prec=prec )+
r"""
Tennis players~\cite{tennis2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['Tennis_ATP'], prec=prec )+
r"""
Golf players~\cite{golf2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['Golf_OWGR'], prec=prec )+
r"""
Football scorers~\cite{football2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['Football_Scorers'], prec=prec )+
r"""
NASCAR drivers (Busch)~\cite{nascar2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['Nascar_BuschGrandNational'], prec=prec )+
r"""
NASCAR drivers (Winston Cup)~\cite{nascar2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['Nascar_WinstonCupGrandNational'], prec=prec )+
r"""
National football teams~\cite{teams2018}"""
+r' & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} & {:.{prec}f} \\'.format( *params_model_tuples['Football_FIFA'], prec=prec )+
r"""
\bottomrule
\end{tabular}}
\caption{\small {\bf Data measures and fitted model parameters}. Values of empirical measures used in the fitting process (mean flux $F$, average openness derivative $\langle \dot{o} \rangle$, and success $S^{++}$) and of fitted model parameters (relative ranking list size $p_0$, displacement probability $p_{\tau}$, and replacement probability $p_{\nu}$) for all considered datasets (see \tref{tab:datasets} and \sref{sec:data}). For open ranking lists, we fit the model to empirical data by setting $N = N_{T-1}$ (i.e. $p_0 = N_0 / N_{T-1}$, see \sref{ssec:modelDef}) and by computing $p_{\tau}$ and $p_{\nu}$ numerically from \eref{eq:paramsSystEqs} in terms of $F$ and $\langle \dot{o} \rangle$ (\fref{fig:fitting}). For closed ranking lists, we set $N = N_0$ and $p_{\nu} = 0$, and obtain $p_{\tau}$ explicitly from \eref{eq:fittingClosedSyst} in terms of $S^{++}$ (for lag $l=1$ and threshold $c = 0.5$). Datasets are classified by an (arbitrary) system type based on the nature of the elements in the ranking list.}
\label{tab:parameters}
\end{table}
"""
	)

#! /usr/bin/env python

### SCRIPT FOR CREATING MODEL TABLE IN FARRANKS PROJECT ###

#import modules
import os, sys
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc


## RUNNING TABLE SCRIPT ##

if __name__ == "__main__":

	## CONF ##
	ntimes_small, ntimes_big = 10, 100 #number(s) of model realisations
	prop_names = [ 'flux_time', 'openness', 'flux_rank', 'change', 'diversity', 'success' ] #properties to fit

	#locations
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_data = root_loc+'nullModel/v4/files/' #location of output files
	saveloc_model = root_loc+'nullModel/v4/files/model/'

	#datasets to explore
	open_dsets = [ 'AcademicRanking', 'AtlasComplex', 'Citations', 'Cities_RU', 'english', 'enron-sent-mails-weekly', 'FIDEFemale', 'FIDEMale', 'Football_FIFA', 'Football_Scorers', 'Fortune', 'french', 'german', 'Golf_OWGR', 'Hienas', 'italian', 'Nascar_BuschGrandNational', 'Nascar_WinstonCupGrandNational', 'Poker_GPI', 'russian', 'spanish', 'Tennis_ATP', 'TheGuardian_numberComments' ]
	closed_dsets = [ 'Cities_UK', 'Earthquakes_avgMagnitude', 'Earthquakes_numberQuakes', 'metroMex', 'UndergroundByWeek' ] #closed datasets
	ignored_dsets = [ 'TheGuardian_avgRecommends', 'github-watch-weekly' ] #ignored datasets (while we run better sims!)

#	datasets = [ 'VideogameEarnings', 'Virus' ] #shady data

	#get model parameters for all datasets

	params_model_tuples = {} #initialise dict of model params tuples
	for dataname in open_dsets+closed_dsets+ignored_dsets: #loop through considered datasets

		#get model parameters for selected dataset
		if dataname == 'AcademicRanking': #set model realisations (dataset-specific!)
			params = { 'ntimes' : ntimes_big }
		else:
			params = { 'ntimes' : ntimes_small }
		params_model = data_misc.data_estimate_params( dataname, params, 'y', saveloc_data, saveloc_model )

		params_model_tuples[ dataname ] = [] #initialise tuple as list
		for prop_name in prop_names: #loop through rank measures to fit

			#add parameters for open systems
			if dataname in open_dsets:

				#first, handle data not yet ready
				if prop_name == 'success' and dataname in [ 'TheGuardian_numberComments', 'english', 'french', 'german', 'italian', 'russian', 'spanish', 'Poker_GPI', 'Tennis_ATP', 'FIDEFemale', 'FIDEMale', 'Golf_OWGR' ]:
					params_model_tuples[ dataname ].extend( [ '', '' ] )

				#then systems with ready data
				else:
					params_model_tuples[ dataname ].extend( params_model.loc[ prop_name, [ 'ptau', 'pnu' ] ] )

			#add parameters for closed systems
			if dataname in closed_dsets:
				if prop_name in [ 'flux_time', 'openness', 'flux_rank' ]:
					params_model_tuples[ dataname ].extend( [ '--', '--' ] )
				else:
					params_model_tuples[ dataname ].extend( params_model.loc[ prop_name, [ 'ptau', 'pnu' ] ] )

			#add text for ignored systems (data not yet ready):
			if dataname in ignored_dsets:
				params_model_tuples[ dataname ].extend( [ '', '' ] )

		params_model_tuples[ dataname ] = tuple( params_model_tuples[ dataname ] ) #go to tuple!

	## PRINTING ##

	print(
r"""
\begin{table}[!ht]
\small
\noindent\makebox[\textwidth]{
\begin{tabular}{l | r r | r r | r r | r r | r r | r r}
\toprule
& \multicolumn{12}{c}{Measure} \\
& \multicolumn{2}{c}{$F_t$} & \multicolumn{2}{c}{$o_t$} & \multicolumn{2}{c}{$F^-_R$} & \multicolumn{2}{c}{$C_R$} & \multicolumn{2}{c}{$d_R$} & \multicolumn{2}{c}{$S^{++}_l$} \\
\cmidrule(l){2-13}
Dataset & $p_{\tau}$ & $p_{\nu}$ & $p_{\tau}$ & $p_{\nu}$ & $p_{\tau}$ & $p_{\nu}$ & $p_{\tau}$ & $p_{\nu}$ & $p_{\tau}$ & $p_{\nu}$ & $p_{\tau}$ & $p_{\nu}$ \\
\midrule
{\bf Society} & & & & & & & & & & & & \\
\cmidrule(l){1-1}
GitHub repositories~\cite{github2018}"""
+r' & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\'.format( *params_model_tuples['github-watch-weekly'] )+
r"""
The Guardian readers (recc)~\cite{guardian2018}"""
+r' & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\'.format( *params_model_tuples['TheGuardian_avgRecommends'] )+
r"""
The Guardian readers (comm)~\cite{guardian2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {} & {} \\'.format( *params_model_tuples['TheGuardian_numberComments'] )+
r"""
Enron emails~\cite{enron2015}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['enron-sent-mails-weekly'] )+
r"""
Scientists~\cite{sinatra2015}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['Citations'] )+
r"""
Universities~\cite{shanghai2016}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['AcademicRanking'] )+
r"""
\midrule
{\bf Languages} & & & & & & & & & & & & \\
\cmidrule(l){1-1}
Russian~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {} & {} \\'.format( *params_model_tuples['russian'] )+
r"""
Spanish~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {} & {} \\'.format( *params_model_tuples['spanish'] )+
r"""
German~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {} & {} \\'.format( *params_model_tuples['german'] )+
r"""
French~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {} & {} \\'.format( *params_model_tuples['french'] )+
r"""
Italian~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {} & {} \\'.format( *params_model_tuples['italian'] )+
r"""
English~\cite{google2018,michel2011,cocho2015,morales2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {} & {} \\'.format( *params_model_tuples['english'] )+
r"""
\midrule
{\bf Economics} & & & & & & & & & & & & \\
\cmidrule(l){1-1}
Companies~\cite{fortune2005}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['Fortune'] )+
r"""
Countries~\cite{atlas2018,hidalgo2009,hausmann2014}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['AtlasComplex'] )+
r"""
\midrule
{\bf Infrastructure} & & & & & & & & & & & & \\
\cmidrule(l){1-1}
Cities (RU)~\cite{cottineau2016}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['Cities_RU'] )+
r"""
Metro stations (London)~\cite{murcio2015}"""
+r' & {} & {} & {} & {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['UndergroundByWeek'] )+
r"""
Cities (GB)~\cite{edwards2016}"""
+r' & {} & {} & {} & {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['Cities_UK'] )+
r"""
Metro stations (Mexico)"""
+r' & {} & {} & {} & {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['metroMex'] )+
r"""
\midrule
{\bf Nature} & & & & & & & & & & & & \\
\cmidrule(l){1-1}
Hyenas~\cite{ilany2015}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['Hienas'] )+
r"""
Regions JP (quake mag)~\cite{junec2018,karsai2012}"""
+r' & {} & {} & {} & {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['Earthquakes_avgMagnitude'] )+
r"""
Regions JP (quakes)~\cite{junec2018,karsai2012}"""
+r' & {} & {} & {} & {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['Earthquakes_numberQuakes'] )+
r"""
\midrule
{\bf Sports} & & & & & & & & & & & & \\
\cmidrule(l){1-1}
Chess players (male)~\cite{fide2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {} & {} \\'.format( *params_model_tuples['FIDEMale'] )+
r"""
Chess players (female)~\cite{fide2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {} & {} \\'.format( *params_model_tuples['FIDEFemale'] )+
r"""
Poker players~\cite{poker2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {} & {} \\'.format( *params_model_tuples['Poker_GPI'] )+
r"""
Tennis players~\cite{tennis2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {} & {} \\'.format( *params_model_tuples['Tennis_ATP'] )+
r"""
Golf players~\cite{golf2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {} & {} \\'.format( *params_model_tuples['Golf_OWGR'] )+
r"""
Football scorers~\cite{football2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['Football_Scorers'] )+
r"""
NASCAR drivers (Busch)~\cite{nascar2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['Nascar_BuschGrandNational'] )+
r"""
NASCAR drivers (Winston Cup)~\cite{nascar2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['Nascar_WinstonCupGrandNational'] )+
r"""
National football teams~\cite{teams2018}"""
+r' & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( *params_model_tuples['Football_FIFA'] )+
r"""
\bottomrule
\end{tabular}}
\caption{{\bf Model parameters for datasets used in this study}. Values of the model parameters $p_{\tau}$ and $p_{\nu}$ for each of the considered datasets (see \tref{tab:datasets} and \sref{sec:data}), obtained by fitting the open/closed model of \sref{sec:model} to data with non-zero/zero mean flux $F$, respectively. The fitting process is a grid search of parameter space $(p_{\tau}, p_{\nu}) \in [0, 1] \times [0, 1]$ (for the open model) or $p_{\tau} \in [0, 1]$ with $p_{\nu} = 0$ (for the closed model). The performance metric is the mean squared error between data and model of one of six rank measures: rank flux $F_t$, rank openness $o_t$, rank out-flux $F^-_R$, rank change $C_R$, rank diversity $d_R$, and success $S^{++}_l$. For closed ranking lists, model fitting is not performed for $F_t$, $o_t$ and $F^-_R$ (dashes). Datasets are classified by an (arbitrary) system type based on the nature of the elements in the ranking list.}
\label{tab:parameters}
\end{table}
"""
	)

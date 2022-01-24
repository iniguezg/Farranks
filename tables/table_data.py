#! /usr/bin/env python

#Farranks - Exploring rank dynamics in complex systems
#Copyright (C) 2022 Gerardo Iñiguez

### SCRIPT FOR CREATING DATA TABLE IN FARRANKS PROJECT ###

#import modules
import os, sys
import pandas as pd
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc


## RUNNING TABLE SCRIPT ##

if __name__ == "__main__":

	## CONF ##
	root_loc = expanduser('~') + '/prg/xocial/Farranks/' #root location
	saveloc_data = root_loc+'nullModel/v4/files/' #location of output files

	#get parameters for all datasets
	params_data = pd.read_pickle( saveloc_data+'params_data.pkl' )
	intervals_data = pd.read_pickle( saveloc_data+'intervals_data.pkl' )

	## PRINTING ##

	print(
r"""
\begin{table}[!ht]
\small
\noindent\makebox[\textwidth]{
\begin{tabular}{l l l | r r r r}
\toprule
& & & \multicolumn{4}{c}{Measure} \\
\cmidrule(l){4-7}
Dataset & Element & Score & $N_{T - 1}$ & $N_0$ & $T$ & $\ell$ (days) \\
\midrule
{\bf Society} & & & & & & \\
\cmidrule(l){1-1}
GitHub repositories~\cite{github2018} & repository & \# watchers &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['github-watch-weekly', 'N'], params_data.loc['github-watch-weekly', 'N0'], params_data.loc['github-watch-weekly', 'T'], intervals_data.loc['github-watch-weekly', 'tdays'] )+
r"""
The Guardian readers (recc)~\cite{guardian2018} & person & avg \# recommends &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['TheGuardian_avgRecommends', 'N'], params_data.loc['TheGuardian_avgRecommends', 'N0'], params_data.loc['TheGuardian_avgRecommends', 'T'], intervals_data.loc['TheGuardian_avgRecommends', 'tdays'] )+
r"""
The Guardian readers (comm)~\cite{guardian2018} & person & \# comments &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['TheGuardian_numberComments', 'N'], params_data.loc['TheGuardian_numberComments', 'N0'], params_data.loc['TheGuardian_numberComments', 'T'], intervals_data.loc['TheGuardian_numberComments', 'tdays'] )+
r"""
Enron emails~\cite{enron2015} & person & \# emails &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['enron-sent-mails-weekly', 'N'], params_data.loc['enron-sent-mails-weekly', 'N0'], params_data.loc['enron-sent-mails-weekly', 'T'], intervals_data.loc['enron-sent-mails-weekly', 'tdays'] )+
r"""
Scientists~\cite{sinatra2015,sinatra2016} & person & \# citations &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['Citations', 'N'], params_data.loc['Citations', 'N0'], params_data.loc['Citations', 'T'], intervals_data.loc['Citations', 'tdays'] )+
r"""
Universities~\cite{shanghai2016} & university & ARWU score~\cite{shanghai2018} &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['AcademicRanking', 'N'], params_data.loc['AcademicRanking', 'N0'], params_data.loc['AcademicRanking', 'T'], intervals_data.loc['AcademicRanking', 'tdays'] )+
r"""
\midrule
{\bf Languages} & & & & & & \\
\cmidrule(l){1-1}
Russian~\cite{google2018,michel2011,cocho2015,morales2018} & word & frequency &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['russian', 'N'], params_data.loc['russian', 'N0'], params_data.loc['russian', 'T'], intervals_data.loc['russian', 'tdays'] )+
r"""
Spanish~\cite{google2018,michel2011,cocho2015,morales2018} & word & frequency &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['spanish', 'N'], params_data.loc['spanish', 'N0'], params_data.loc['spanish', 'T'], intervals_data.loc['spanish', 'tdays'] )+
r"""
German~\cite{google2018,michel2011,cocho2015,morales2018} & word & frequency &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['german', 'N'], params_data.loc['german', 'N0'], params_data.loc['german', 'T'], intervals_data.loc['german', 'tdays'] )+
r"""
French~\cite{google2018,michel2011,cocho2015,morales2018} & word & frequency &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['french', 'N'], params_data.loc['french', 'N0'], params_data.loc['french', 'T'], intervals_data.loc['french', 'tdays'] )+
r"""
Italian~\cite{google2018,michel2011,cocho2015,morales2018} & word & frequency &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['italian', 'N'], params_data.loc['italian', 'N0'], params_data.loc['italian', 'T'], intervals_data.loc['italian', 'tdays'] )+
r"""
English~\cite{google2018,michel2011,cocho2015,morales2018} & word & frequency &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['english', 'N'], params_data.loc['english', 'N0'], params_data.loc['english', 'T'], intervals_data.loc['english', 'tdays'] )+
r"""
\midrule
{\bf Economics} & & & & & & \\
\cmidrule(l){1-1}
Companies~\cite{fortune2005} & company & revenue &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['Fortune', 'N'], params_data.loc['Fortune', 'N0'], params_data.loc['Fortune', 'T'], intervals_data.loc['Fortune', 'tdays'] )+
r"""
Countries~\cite{atlas2018,hidalgo2009,hausmann2014} & country & complexity~\cite{atlas2018,hidalgo2009,hausmann2014} &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['AtlasComplex', 'N'], params_data.loc['AtlasComplex', 'N0'], params_data.loc['AtlasComplex', 'T'], intervals_data.loc['AtlasComplex', 'tdays'] )+
r"""
\midrule
{\bf Infrastructure} & & & & & & \\
\cmidrule(l){1-1}
Cities (RU)~\cite{cottineau2016} & city & population &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['Cities_RU', 'N'], params_data.loc['Cities_RU', 'N0'], params_data.loc['Cities_RU', 'T'], intervals_data.loc['Cities_RU', 'tdays'] )+
r"""
Metro stations (London)~\cite{murcio2015} & station & \# passengers &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['UndergroundByWeek', 'N'], params_data.loc['UndergroundByWeek', 'N0'], params_data.loc['UndergroundByWeek', 'T'], 0.01 )+
r"""
Cities (GB)~\cite{edwards2016} & city & population &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['Cities_UK', 'N'], params_data.loc['Cities_UK', 'N0'], params_data.loc['Cities_UK', 'T'], intervals_data.loc['Cities_UK', 'tdays'] )+
"""
Metro stations (Mexico) & station & \# passengers &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['metroMex', 'N'], params_data.loc['metroMex', 'N0'], params_data.loc['metroMex', 'T'], intervals_data.loc['metroMex', 'tdays'] )+
r"""
\midrule
{\bf Nature} & & & & & & \\
\cmidrule(l){1-1}
Hyenas~\cite{ilany2015} & animal & association index &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['Hienas', 'N'], params_data.loc['Hienas', 'N0'], params_data.loc['Hienas', 'T'], intervals_data.loc['Hienas', 'tdays'] )+
r"""
Regions JP (quake mag)~\cite{junec2018,karsai2012} & region & avg quake magnitude &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['Earthquakes_avgMagnitude', 'N'], params_data.loc['Earthquakes_avgMagnitude', 'N0'], params_data.loc['Earthquakes_avgMagnitude', 'T'], intervals_data.loc['Earthquakes_avgMagnitude', 'tdays'] )+
r"""
Regions JP (quakes)~\cite{junec2018,karsai2012} & region & \# quakes &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['Earthquakes_numberQuakes', 'N'], params_data.loc['Earthquakes_numberQuakes', 'N0'], params_data.loc['Earthquakes_numberQuakes', 'T'], intervals_data.loc['Earthquakes_numberQuakes', 'tdays'] )+
r"""
\midrule
{\bf Sports} & & & & & & \\
\cmidrule(l){1-1}
Chess players (male)~\cite{fide2018} & person & Elo rating~\cite{elo2018} &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['FIDEMale', 'N'], params_data.loc['FIDEMale', 'N0'], params_data.loc['FIDEMale', 'T'], intervals_data.loc['FIDEMale', 'tdays'] )+
r"""
Chess players (female)~\cite{fide2018} & person & Elo rating~\cite{elo2018} &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['FIDEFemale', 'N'], params_data.loc['FIDEFemale', 'N0'], params_data.loc['FIDEFemale', 'T'], intervals_data.loc['FIDEFemale', 'tdays'] )+
r"""
Poker players~\cite{poker2018} & person & GPI score~\cite{gpi2018} &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['Poker_GPI', 'N'], params_data.loc['Poker_GPI', 'N0'], params_data.loc['Poker_GPI', 'T'], intervals_data.loc['Poker_GPI', 'tdays'] )+
r"""
Tennis players~\cite{tennis2018} & person & ATP points~\cite{atp2018} &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['Tennis_ATP', 'N'], params_data.loc['Tennis_ATP', 'N0'], params_data.loc['Tennis_ATP', 'T'], intervals_data.loc['Tennis_ATP', 'tdays'] )+
r"""
Golf players~\cite{golf2018} & person & OWGR points~\cite{owgr2018} &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['Golf_OWGR', 'N'], params_data.loc['Golf_OWGR', 'N0'], params_data.loc['Golf_OWGR', 'T'], intervals_data.loc['Golf_OWGR', 'tdays'] )+
r"""
Football scorers~\cite{football2018} & person & \# goals~\cite{fwr2018} &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['Football_Scorers', 'N'], params_data.loc['Football_Scorers', 'N0'], params_data.loc['Football_Scorers', 'T'], intervals_data.loc['Football_Scorers', 'tdays'] )+
r"""
NASCAR drivers (Busch)~\cite{nascar2018} & person & NASCAR points &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['Nascar_BuschGrandNational', 'N'], params_data.loc['Nascar_BuschGrandNational', 'N0'], params_data.loc['Nascar_BuschGrandNational', 'T'], intervals_data.loc['Nascar_BuschGrandNational', 'tdays'] )+
r"""
NASCAR drivers (Winston Cup)~\cite{nascar2018} & person & NASCAR points &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['Nascar_WinstonCupGrandNational', 'N'], params_data.loc['Nascar_WinstonCupGrandNational', 'N0'], params_data.loc['Nascar_WinstonCupGrandNational', 'T'], intervals_data.loc['Nascar_WinstonCupGrandNational', 'tdays'] )+
r"""
National football teams~\cite{teams2018} & team & FIFA points~\cite{fifa2018} &"""
+r' {} & {} & {} & {:.2f} \\'.format( params_data.loc['Football_FIFA', 'N'], params_data.loc['Football_FIFA', 'N0'], params_data.loc['Football_FIFA', 'T'], intervals_data.loc['Football_FIFA', 'tdays'] )+
r"""
\bottomrule
\end{tabular}}
\caption{\small {\bf Datasets used in this study}. Characteristics of the available datasets, including the observed system size $N_{T - 1}$, ranking list size $N_0$, number of observations $T$, and real time interval between observations $\langle \tau_{\mathrm{days}} \rangle$. The table includes an (arbitrary) system type based on the nature of the elements in the ranking list, as well as the corresponding definitions of elements and scores for each system.}
\label{tab:datasets}
\end{table}
"""
	)

#!/bin/bash

module load anaconda

# for dataname in AcademicRanking AtlasComplex Citations Cities_RU english enron-sent-mails-weekly FIDEFemale FIDEMale Football_Scorers Fortune french german github-watch-weekly Golf_OWGR Hienas italian Nascar_BuschGrandNational Nascar_WinstonCupGrandNational Poker_GPI russian spanish Tennis_ATP TheGuardian_avgRecommends TheGuardian_numberComments
# for dataname in english french german github-watch-weekly italian russian spanish
for dataname in AtlasComplex Citations Cities_RU enron-sent-mails-weekly Football_Scorers Fortune Hienas Nascar_BuschGrandNational Nascar_WinstonCupGrandNational Tennis_ATP
do
	sbatch -J $dataname -o $dataname.out script_fitData.sh $dataname
done

# sbatch -J Citations -o Citations.out script_fitData.sh Citations
#
# sbatch -J english -o english.out script_fitData.sh english
#
# sbatch -J FIDEFemale -o FIDEFemale.out script_fitData.sh FIDEFemale
#
# sbatch -J FIDEMale -o FIDEMale.out script_fitData.sh FIDEMale
#
# sbatch -J french -o french.out script_fitData.sh french
#
# sbatch -J german -o german.out script_fitData.sh german
#
# sbatch -J github-watch-weekly -o github-watch-weekly.out script_fitData.sh github-watch-weekly
#
# sbatch -J Golf_OWGR -o Golf_OWGR.out script_fitData.sh Golf_OWGR
#
# sbatch -J Hienas -o Hienas.out script_fitData.sh Hienas
#
# sbatch -J italian -o italian.out script_fitData.sh italian
#
# sbatch -J metroMex -o metroMex.out script_fitData.sh metroMex
#
# sbatch -J Poker_GPI -o Poker_GPI.out script_fitData.sh Poker_GPI
#
# sbatch -J russian -o russian.out script_fitData.sh russian
#
# sbatch -J spanish -o spanish.out script_fitData.sh spanish
#
# sbatch -J Tennis_ATP -o Tennis_ATP.out script_fitData.sh Tennis_ATP
#
# sbatch -J TheGuardian_avgRecommends -o TheGuardian_avgRecommends.out script_fitData.sh TheGuardian_avgRecommends
#
# sbatch -J TheGuardian_numberComments -o TheGuardian_numberComments.out script_fitData.sh TheGuardian_numberComments

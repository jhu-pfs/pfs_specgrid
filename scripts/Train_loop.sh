#!/bin/bash
#for para in "T_eff" "Fe_H" "log_g" 
#do
#	for name in 'v0116'
#	do
#		./scripts/convert.sh sbatch -p elephant --cpus 48 --mem 128G bosz basic --in $PFSSPEC_DATA/import/stellar/grid/bosz --out $PFSSPEC_DATA/train/stellar/grid/bosz_100k_6300_9700_ext006_"$para"_"$name"_test --sample-mode random --sample-dist beta --sample-count 100000 --interp-mode spline --interp-param $para --wave 6300 9700 --wave-bins 1200 --wave-log --T_eff 4000 8000 --Fe_H -2.5 0.75 --log_g 0 5 --ext-random -2.56 0.5 --redden --norm mean --noise $PFSSPEC_DATA/subaru/pfs/noise/sim8hr.dat   
#	done
#	wait
#done
#wait
for para in "T_eff" "Fe_H" "log_g" 
do
	for name in 'v0116_norm'
	do
		./scripts/convert.sh sbatch -p elephant --cpus 48 --mem 128G bosz basic --in $PFSSPEC_DATA/import/stellar/grid/bosz --out $PFSSPEC_DATA/train/stellar/grid/bosz_100k_6300_9700_ext006_"$para"_"$name" --sample-mode random --sample-dist beta --sample-count 100000 --interp-mode spline --interp-param $para --wave 6300 9700 --wave-bins 1200 --wave-log --T_eff 4000 8000 --Fe_H -2.5 0.75 --log_g 0 5 --ext-random -2.56 0.5 --redden --cont chebyshev --cont-wave 6300 9700 --norm cont  
	done
	wait
done
wait

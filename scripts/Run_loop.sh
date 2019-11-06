#!/bin/bash
#./scripts/NN_params.sh
#bosz_100k_6300_9700_ext_"$para"_"$input"
#bosz_6300_9700_100k_8h_ext_interp_rand_norm_v1030
# --opt sgd --lr-sch drop --lr $ini_lr 10 0.5

for ep in 1
do
	for para in 'Fe_H' 'log_g'
	# 'T_eff' 'Fe_H' 'log_g' 'extinction' 
	do
		for unit in 128
		do
			for level in 14 
			do
				for ini_lr in 0.01
				do
					for v_num in 'v0116'
					#'v0116' 'v0116_norm'
					do
					./scripts/train.sh sbatch  -p v100 --gpus 1 reg kurucz dense --in /scratch/ceph/dobos/data/pfsspec/train/stellar/grid/bosz_100k_6300_9700_ext006_"$para"_"$v_num" --out /scratch/ceph/dobos/data/pfsspec/run/bosz --labels "$para" --levels $level --units $unit --split 0.3 --epochs $ep --batch 16 --patience 500 --act relu --loss mse --opt adam --lr $ini_lr --dropout 0 --aug no --noiz 0.1 --no-batchnorm --name Viska_"$para"_"$v_num"_test/ 
					done
					wait
				done
				wait
			done
			wait
		done
		wait	   
	done
	wait
done
wait
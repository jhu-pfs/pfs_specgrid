#!/bin/bash
for ep in 10
do
	for para in 'T_eff' 'Fe_H' 'log_g' 'extinction'
	do
		for unit in 256 2048
		do
			for level in 14 22
			do
				for ini_lr in 0.001 0.003 0.01 0.03
				do
				./scripts/train.sh sbatch  -p v100 --gpus 1 reg kurucz dense --in /scratch/ceph/dobos/data/pfsspec/train/stellar/grid/bosz_6300_9700_100k_8h_ext_interp_rand_v1178 --out /scratch/ceph/dobos/data/pfsspec/run/bosz --labels $para --levels $level --units $unit --split 0.3 --epochs $ep --batch 16 --patience 500 --act relu --loss mse --opt sgd --lr-sch drop --lr $ini_lr 50 0.5 --dropout 0 --aug no --noiz 0.1 --no-batchnorm --name Viska_$para_$ep/ext1178n01/$level_$unit/sgd/$ini_lr
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
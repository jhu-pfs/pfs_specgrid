#!/bin/bash
for i in 256 512
do
   ./scripts/train.sh sbatch  -p v100 --gpus 1 reg kurucz dense --in /scratch/ceph/dobos/data/pfsspec/train/stellar/grid/bosz_6300_9700_100k_8h_ext_interp_rand_v1178 --out /scratch/ceph/dobos/data/pfsspec/run/bosz --labels T_eff/10000 --levels 14 --units $i --split 0.3 --epochs 10 --batch 16 --patience 500 --act relu --loss mse --opt adam --dropout 0 --aug no --noiz 0.1 --no-batchnorm --name ext1178n01/adam/14_$i
done
wait
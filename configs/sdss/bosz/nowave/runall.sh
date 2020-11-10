JOBNAME=test411
RUNDIR=train/sdss_stellar_model

GPU_PARTITION=v100
CPU_PARTITION=elephant

PREPARE_MEM=512G
PREPARE_CPU=64

TRAIN_MEM=24G
TRAIN_CPU=12

function prepare_model {
    DATASET=$1
    ./scripts/prepare.sh sbatch -p $CPU_PARTITION --mem $PREPARE_MEM --cpus $PREPARE_CPU --time 2-0:0:0 \
        model bosz sdss \
        --config ./configs/sdss/bosz/nowave/prepare/$DATASET.json \
                 ./configs/sdss/bosz/nowave/instrument.json \
}

functin prepare_segue {
    #
}

function predict_segue {
    PARAM=$1
    ./scripts/predict.sh reg dense sdss \
        --model ${PFSSPEC_DATA}/$RUNDIR/run/$JOBNAME-1M-drop_dense_16_2048_mse_sgd_$PARAM \
        --in ${PFSSPEC_DATA}/$RUNDIR/dataset/segue/nowave/test \
        --out ${PFSSPEC_DATA}/$RUNDIR/predict/segue/$JOBNAME_$PARAM \
        --labels $PARAM
}

# Prepare training, validation and test sets
prepare_model train
prepare_model valid
prepare_model test_Fe_H
prepare_model test_log_g
prepare_model test_log_g

# Prepare SEGUE to feed into the network
prepare_segue segue

# Predict on SDSS SEGUE
predict_segue T_eff
predict_segue Fe_H
predict_segue log_g

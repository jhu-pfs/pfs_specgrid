#!/bin/bash

# Work around issues with saving weights when running on multiple threads
export HDF5_USE_FILE_LOCKING=FALSE

PARAMS=""
COMMAND="$1"
shift

if [[ $1 == "sbatch" ]] || [[ $1 == "srun" ]]; then
    RUNMODE=$1
    shift

    SBATCH_PARTITION=default
    SBATCH_MEM=16G
    SBATCH_GPUS=0
    SBATCH_CPUS_PER_TASK=8

    while (( "$#" )); do
        case "$1" in
            -p|--partition)
                SBATCH_PARTITION=$2
                shift 2
                ;;
            --mem)
                SBATCH_MEM=$2
                shift 2
                ;;
            -G|--gpus)
                SBATCH_GPUS=$2
                shift 2
                ;;
            -c|--cpus-per-task|--cpus)
                SBATCH_CPUS_PER_TASK=$2
                shift 2
                ;;
            --) # end argument parsing
                shift
                break
                ;;
            #-*|--*=) # unsupported flags
                #  echo "Error: Unsupported flag $1" >&2
                #  exit 1
                #  ;;
            *) # preserve all other arguments
                PARAMS="$PARAMS $1"
                shift
                ;;
        esac
    done
elif [[ $1 == "run" ]]; then
    RUNMODE=$1
    shift
else
    RUNMODE="run"
fi

if [[ $RUNMODE == "run" ]]; then
    exec $COMMAND $@
elif [[ $RUNMODE == "srun" ]]; then
    exec srun --partition $SBATCH_PARTITION --gpus $SBATCH_GPUS \
              --cpus-per-task $SBATCH_CPUS_PER_TASK --mem $SBATCH_MEM \
              $COMMAND $PARAMS
elif [[ $RUNMODE == "sbatch" ]]; then
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition $SBATCH_PARTITION
#SBATCH --gpus $SBATCH_GPUS
#SBATCH --cpus-per-task $SBATCH_CPUS_PER_TASK
#SBATCH --mem $SBATCH_MEM

set -e

out=slurm-\$SLURM_JOB_ID.out
srun $COMMAND $PARAMS
outdir=\$(cat \$out | grep -Po 'Output directory is (\K.+)')
mv \$out \$outdir/slurm.out
EOF
else
    echo "Invalid RUNMODE: $RUNMODE"
    exit -1
fi
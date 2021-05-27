#!/bin/bash

# Fit the upper envelope of synthetic stellar spectra and fit with RBF
# It needs to be run with a bunch of config files in a single config directory
# or with command line arguments passed in.

# Examples:
#   ./scripts/build_rbf.sh grid bosz
#   ./scripts/build_rbf.sh grid bosz --config ./configs/import/bosz/small --in /.../grid/bosz_5000/ --out /.../rbf/bosz_5000

set -e

# Process command-line arguments

PYTHON_DEBUG=0

CONFIGDIR="./configs/import/bosz/small"
INDIR="${PFSSPEC_DATA}/import/stellar/grid/bosz_5000"
OUTDIR="${PFSSPEC_DATA}/import/stellar/rbf/bosz_5000"
PARAMS=""

TYPE="$1"
SOURCE="$2"
shift 2

while (( "$#" )); do
    case "$1" in
        --debug)
            PYTHON_DEBUG=1
            shift
            ;;
        --config)
            CONFIGDIR="$2"
            shift 2
            ;;
        --in)
            INDIR="$2"
            shift 2
            ;;
        --out)
            OUTDIR="$2"
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

if [[ -d "$OUTDIR/fit" ]]; then
    echo "Skipping fitting upper envelope."
else
    echo "Fitting upper envelope..."
    ./scripts/fit.sh $TYPE $SOURCE \
        --config "$CONFIGDIR/common.json" \
        --in "$INDIR" --out "$OUTDIR/fit" \
        --step fit $PARAMS
fi

if [[ -d "$OUTDIR/fill" ]]; then
    echo "Skipping filling in holes / smoothing."
else
    echo "Filling in holes / smoothing..."
    ./scripts/fit.sh $TYPE $SOURCE \
        --config "$CONFIGDIR/common.json" \
        --in "$INDIR" --out "$OUTDIR/fill" \
        --params "$OUTDIR/fit" \
        --step fill $PARAMS
fi

if [[ -d "$OUTDIR/fitrbf" ]]; then
    echo "Skipping RBF on continuum parameters."
else
    echo "Running RBF on continuum parameters..."
    ./scripts/rbf.sh $TYPE $SOURCE \
        --config "$CONFIGDIR/common.json" "$CONFIGDIR/rbf.json" \
        --in "$INDIR" --out "$OUTDIR/fitrbf" \
        --params "$OUTDIR/fill" \
        --step fit $PARAMS
fi

if [[ -d "$OUTDIR/norm" ]]; then
    echo "Skipping normalizing spectra."
else
    echo "Normalizing spectra..."
    ./scripts/fit.sh $TYPE $SOURCE \
        --config "$CONFIGDIR/common.json" \
        --in "$INDIR" --out "$OUTDIR/norm" \
        --params "$OUTDIR/fitrbf" --rbf \
        --step norm $PARAMS
fi

if [[ -d "$OUTDIR/pca" ]]; then
    echo "Skipping PCA."
else
    echo "Running PCA..."
    ./scripts/pca.sh $TYPE $SOURCE \
        --config "$CONFIGDIR/common.json" "$CONFIGDIR/pca.json" \
        --in "$OUTDIR/norm" --out "$OUTDIR/pca" \
        --params "$OUTDIR/fitrbf" --rbf $PARAMS
fi

if [[ -d "$OUTDIR/rbf" ]]; then
    echo "Skipping RBF on principal components."
else
    echo "Running RBF on principal components..."
    ./scripts/rbf.sh $TYPE $SOURCE \
        --config "$CONFIGDIR/common.json" "$CONFIGDIR/rbf.json" \
        --in "$OUTDIR/pca" --out "$OUTDIR/rbf" \
        --params "$OUTDIR/fitrbf" --rbf --pca \
        --step pca $PARAMS
fi
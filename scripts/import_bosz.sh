#!/bin/bash

set -o noglob
./scripts/run.sh "-m pfsspec.scripts.import_bosz" $@
set +o noglob
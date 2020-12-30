#!/bin/bash

set -o noglob
./scripts/run.sh "-m pfsspec.scripts.import_" $@
set +o noglob
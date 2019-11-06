#!/bin/bash

set -o noglob
exec python -m pfsspec.scripts.import_bosz $@
set +o noglob
#!/bin/bash

set -o noglob
exec python -m pfsspec.scripts.import_psf $@
set +o noglob
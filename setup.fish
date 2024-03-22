#!/usr/bin/env fish
set -xg PROJECT_PREFIX (dirname (readlink -m (status --current-filename)))
set -xga PYTHONPATH {$PROJECT_PREFIX}/src
mamba activate diffmet-py311

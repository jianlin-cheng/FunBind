#!/bin/bash
set -e

# Activate conda
source /opt/conda/etc/profile.d/conda.sh
conda activate FunBind

# Run Python with all passed arguments
python funbind_main.py "$@"
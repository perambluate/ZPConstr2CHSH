#!/bin/bash

TOP_DIR=./twoPartyRandomness
PY_FILE=${TOP_DIR}/twoPartyRandomness.py # Specify the file you want to run
LOG_FILE=${TOP_DIR}/log_$(date +%H%M-%d%m%y)        # Save log file as backup or for debugging

# Run in background
nohup python $PY_FILE > $LOG_FILE 2>&1 &

#!/bin/bash

PY_FILE=./bff21/blindRandomness-BFF21.py # Specify the file you want to run
LOG_FILE=log_$(date +%H%M-%d%m%y)        # Save log file as backup or for debugging

# Run in background
nohup python $PY_FILE > $LOG_FILE 2>&1 &

#!/bin/bash

# Script to run ddp.py with all passed arguments, in background, with output redirected to a timestamped nohup log

timestamp=$(date +"%Y%m%d-%H%M%S")
logfile="./logs/nohup-$timestamp"

nohup python3 ddp.py "$@" > "$logfile" 2>&1 &
echo "Script ddp.py started in background. Output is redirected to $logfile"

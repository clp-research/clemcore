#!/bin/bash

# Script to run the benchmark for a single model type (self-play) or single-player games
# For example: run.sh taboo gpt-3

if [ $# -lt 2 ]; then
  echo "Please provide at least two arguments: run.sh <game_name> <player_0> [<player_1>]"
  exit 1
fi

# Load and prepare path
# source venv/bin/activate
source prepare_path.sh

arg_game="$1"
arg_model0="$2"

# Set temperature to 0.0
arg_temp=0.0

if [ $# -eq 2 ]; then
  { time python3 scripts/cli.py run -g "$arg_game" -m "${arg_model0}" -t $arg_temp; } 2>&1 | tee runtime."${arg_game}"."${arg_model0}"--"${arg_model0}".log
fi

if [ $# -eq 3 ]; then
  arg_model1="$3"
  { time python3 scripts/cli.py run -g "$arg_game" -m "${arg_model0}" "${arg_model1}" -t $arg_temp; } 2>&1 | tee runtime."${arg_game}"."${arg_model0}"--"${arg_model1}".log
fi

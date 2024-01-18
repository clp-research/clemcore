#!/bin/bash
# Usage: ./pipeline_llama2_hf.sh
# Preparation: ./setup_hf.sh
# check if needed API key file exists:
if [ -e key.json ]
then
  echo "key.json found."
else
  echo "key.json not found! Loading Llama2 models requires a HuggingFace access token of an account that has Meta's permission to do so."
  exit 1
fi
# activate HF venv:
source venv_hf/bin/activate
source prepare_path.sh
# run pipeline:
echo
echo "==================================================="
echo "PIPELINE: Starting"
echo "==================================================="
echo
game_runs=(
  # Single-player: privateshared
  "privateshared llama-2-13b-chat-hf"
  # Single-player: wordle
  "wordle llama-2-13b-chat-hf"
  # Single-player: wordle_withclue
  "wordle_withclue llama-2-13b-chat-hf"
  # Multi-player taboo
  "taboo llama-2-13b-chat-hf"
  # Multi-player referencegame
  "referencegame llama-2-13b-chat-hf"
  # Multi-player imagegame
  "imagegame llama-2-13b-chat-hf"
  # Multi-player wordle_withcritic
  "wordle_withcritic llama-2-13b-chat-hf"
)
total_runs=${#game_runs[@]}
echo "Number of benchmark runs: $total_runs"
current_runs=1
for run_args in "${game_runs[@]}"; do
  echo "Run $current_runs of $total_runs: $run_args"
  bash -c "./run.sh ${run_args}"
  ((current_runs++))
done
echo "==================================================="
echo "PIPELINE: Finished"
echo "==================================================="
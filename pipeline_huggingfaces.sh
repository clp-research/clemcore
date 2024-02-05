#!/bin/bash
# Usage: ./pipeline_huggingfaces.sh
# Preparation: ./setup_hf.sh
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
  "privateshared koala-13B-HF"
  "privateshared Wizard-Vicuna-13B-Uncensored-HF"
  "privateshared falcon-40b-instruct"
  "privateshared oasst-sft-4-pythia-12b-epoch-3.5"
  # Single-player: wordle
  "wordle koala-13B-HF"
  "wordle Wizard-Vicuna-13B-Uncensored-HF"
  "wordle falcon-40b-instruct"
  "wordle oasst-sft-4-pythia-12b-epoch-3.5"
  # Single-player: wordle_withclue
  "wordle_withclue koala-13B-HF"
  "wordle_withclue Wizard-Vicuna-13B-Uncensored-HF"
  "wordle_withclue falcon-40b-instruct"
  "wordle_withclue oasst-sft-4-pythia-12b-epoch-3.5"
  # Multi-player taboo
  "taboo koala-13B-HF"
  "taboo Wizard-Vicuna-13B-Uncensored-HF"
  "taboo falcon-40b-instruct"
  "taboo oasst-sft-4-pythia-12b-epoch-3.5"
  # Multi-player referencegame
  "referencegame koala-13B-HF"
  "referencegame Wizard-Vicuna-13B-Uncensored-HF"
  "referencegame falcon-40b-instruct"
  "referencegame oasst-sft-4-pythia-12b-epoch-3.5"
  # Multi-player imagegame
  "imagegame koala-13B-HF"
  "imagegame Wizard-Vicuna-13B-Uncensored-HF"
  "imagegame falcon-40b-instruct"
  "imagegame oasst-sft-4-pythia-12b-epoch-3.5"
  # Multi-player wordle_withcritic
  "wordle_withcritic koala-13B-HF"
  "wordle_withcritic Wizard-Vicuna-13B-Uncensored-HF"
  "wordle_withcritic falcon-40b-instruct"
  "wordle_withcritic oasst-sft-4-pythia-12b-epoch-3.5"
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
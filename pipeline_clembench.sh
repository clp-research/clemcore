#!/bin/bash
# Usage: ./pipeline_clembench.sh
# Preparation: ./setup.sh
echo
echo "==================================================="
echo "PIPELINE: Starting"
echo "==================================================="
echo
game_runs=(
  # Single-player: privateshared
  "privateshared text-davinci-003"
  "privateshared gpt-3.5-turbo"
  "privateshared claude-v1.3"
  "privateshared luminous-supreme"
  "privateshared gpt-4"
  # Single-player: wordle
  "wordle text-davinci-003"
  "wordle gpt-3.5-turbo"
  "wordle claude-v1.3"
  "wordle luminous-supreme"
  "wordle gpt-4"
  # Single-player: wordle_withclue
  "wordle_withclue text-davinci-003"
  "wordle_withclue gpt-3.5-turbo"
  "wordle_withclue claude-v1.3"
  "wordle_withclue luminous-supreme"
  "wordle_withclue gpt-4"
  # Multi-player taboo
  "taboo text-davinci-003"
  "taboo gpt-3.5-turbo"
  "taboo claude-v1.3"
  "taboo luminous-supreme"
  "taboo gpt-4"
  "taboo gpt-4 gpt-3.5-turbo"
  "taboo gpt-3.5-turbo gpt-4"
  # Multi-player referencegame
  "referencegame text-davinci-003"
  "referencegame gpt-3.5-turbo-"
  "referencegame claude-v1.3"
  "referencegame luminous-supreme"
  "referencegame gpt-4"
  "referencegame gpt-4 gpt-3.5-turbo"
  "referencegame gpt-3.5-turbo gpt-4"
  # Multi-player imagegame
  "imagegame text-davinci-003"
  "imagegame gpt-3.5-turbo"
  "imagegame claude-v1.3"
  "imagegame luminous-supreme"
  "imagegame gpt-4"
  "imagegame gpt-4 gpt-3.5-turbo"
  "imagegame gpt-3.5-turbo gpt-4"
  # Multi-player wordle_withcritic
  "wordle_withcritic text-davinci-003"
  "wordle_withcritic gpt-3.5-turbo"
  "wordle_withcritic claude-v1.3"
  "wordle_withcritic luminous-supreme"
  "wordle_withcritic gpt-4"
  "wordle_withcritic gpt-4 gpt-3.5-turbo"
  "wordle_withcritic gpt-3.5-turbo gpt-4"
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
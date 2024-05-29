#!/bin/bash

# Script to run referencegame for multiple languages

# Load and prepare path
source prepare_path.sh

game="referencegame"

models=(
#"claude-3-opus-20240229"
#"gpt-4-turbo-2024-04-09"
#"command-r-plus"
#"Llama-3-8b-chat-hf"
#"Llama-3-70b-chat-hf"
#"fsc-openchat-3.5-0106"
#"fsc-codellama-34b-instruct"
)

languages=("de" "en" "it" "ja" "pt" "te" "tk" "tr" "zh") #

results="results/v1.5_multiling"

for lang in "${languages[@]}"; do
  for model in "${models[@]}"; do
    echo "Running ${model} on ${lang}"
    { time python3 scripts/cli.py run -g "${game}" -m "${model}" -i instances_v1.5_"${lang}".json -r "${results}"/"${lang}"; } 2>&1 | tee runtime."${lang}"."${model}".log
  done
  echo "Transcribing ${lang}"
  { time python3 scripts/cli.py transcribe -g "${game}" -r "${results}"/"${lang}"; } 2>&1 | tee runtime.transcribe."${lang}".log
  echo "Scoring ${lang}"
  { time python3 scripts/cli.py score -g "${game}" -r "${results}"/"${lang}"; } 2>&1 | tee runtime.score."${lang}".log
  echo "Evaluating ${lang}"
  { time python3 evaluation/bencheval.py -p "${results}"/"${lang}"; }
done

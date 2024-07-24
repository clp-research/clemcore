#!/bin/bash

# Script to run games in multiple languages
# For each language one subfolder is created: resultspath/lang/model/game...

# Load and prepare path
source prepare_path.sh

games=("referencegame" "imagegame")  # possible games: referencegame, imagegame

models=(
"command-r-plus"
"Llama-3-SauerkrautLM-70b-Instruct"

# "Llama-3-70B-Instruct"
# "Llama-3-8B-Instruct"
# "Mixtral-8x7B-Instruct-v0.1"
# "Mixtral-8x22B-Instruct-v0.1"

# "mock"
)

languages=(
  "en"
  "de" "es" "ru" "te" "tk" "tr"
  "de_google" "es_google" "ru_google" "te_google" "tk_google" "tr_google"
  )

results="results/v1.5_multiling" # "results/v1.5_multiling_liberal"

for lang in "${languages[@]}"; do
  for game in "${games[@]}"; do
    for model in "${models[@]}"; do
      echo "Running ${game} with ${model} in ${lang}"
      { time python3 scripts/cli.py run -g "${game}" -m "${model}" -i instances_v1.5_"${lang}".json -r "${results}"/"${lang}"; } 2>&1 | tee runtime."${game}"."${lang}"."${model}".log
    done
    echo "Transcribing ${game} in ${lang}"
    { time python3 scripts/cli.py transcribe -g "${game}" -r "${results}"/"${lang}"; } 2>&1 | tee runtime.transcribe."${game}"."${lang}".log
    echo "Scoring ${game} in ${lang}"
    { time python3 scripts/cli.py score -g "${game}" -r "${results}"/"${lang}"; } 2>&1 | tee runtime.score."${game}"."${lang}".log
  done
  echo "Evaluating ${lang}"
  { time python3 evaluation/bencheval.py -p "${results}"/"${lang}"; }
done

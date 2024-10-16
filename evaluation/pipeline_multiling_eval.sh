#!/bin/bash
# tested on Linux OpenSUSE Leap 15.4
# Python 3.9.19

# script to run multiling_single_game_evaluation.py with all possible options

source prepare_path.sh

games=("imagegame" "referencegame") # Possible options: "referencegame" "imagegame"
results="results/v1.5_multiling"
# results="results/v1.5_multiling_liberal"

for game in "${games[@]}"; do
    set -x
    python3 evaluation/multiling_single_game_evaluation.py -g "${game}" -p "${results}" -s
    python3 evaluation/multiling_single_game_evaluation.py -g "${game}" -p "${results}" -t human
    python3 evaluation/multiling_single_game_evaluation.py -g "${game}" -p "${results}" -t google
    python3 evaluation/multiling_single_game_evaluation.py -g "${game}" -p "${results}" -d
    python3 evaluation/multiling_single_game_evaluation.py -g "${game}" -p "${results}" -d -t human
    python3 evaluation/multiling_single_game_evaluation.py -g "${game}" -p "${results}" -d -t google
    set +x
done

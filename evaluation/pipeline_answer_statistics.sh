#!/bin/bash
# tested on Linux OpenSUSE Leap 15.4
# Python 3.9.19

# script showing how to run answer_statistics.py

results_dir="results/v1.5_multiling"
output_dir="results/v1.5_multiling/multiling_eval/referencegame/answer_statistics/"
multiling_results_file="results/v1.5_multiling/multiling_eval/referencegame/human+google/v1.5_multiling_referencegame.csv"  # to combine % Played, Quality Score, % Played Consistent and Quality Score Consistent in one table

# run creation of excel overviews (-e) and calculation of statistics (-s) with all options in one call
python3 evaluation/answer_statistics.py "${results_dir}" -o "${output_dir}" -e -s -i -f "${multiling_results_file}"

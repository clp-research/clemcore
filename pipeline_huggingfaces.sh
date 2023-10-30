#!/bin/bash
# Usage: ./pipeline_huggingfaces.sh
# Preparation: ./setup_hf.sh
echo
echo "==================================================="
echo " PIPELINE: Starting $0"
echo "==================================================="
echo
# the key file has to exist
cp key_default.json key.json
# only self-play and single player for now
models=(
  "koala-13B-HF"
  "Wizard-Vicuna-13B-Uncensored-HF"
  "falcon-40b-instruct"
  "oasst-sft-4-pythia-12b-epoch-3.5"
)
source venv_hf/bin/activate
source prepare_path.sh
# Run the benchmark (one model after the other; they consume limited GPU memory)
# but play all games directly (since the loading into memory takes so much time)
for model in "${models[@]}"; do
  { time python3 scripts/cli.py -m "$model" -t 0.0 run all; } 2>&1 | tee runtime.all."$model".log
done
echo "========================================================================="
echo "PIPELINE: Finished"
echo "========================================================================="

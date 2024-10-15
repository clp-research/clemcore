#!/bin/bash
python3 -m venv venv_vllm
source venv_vllm/bin/activate
pip3 install -r requirements.txt
pip3 install -r requirements_vllm.txt
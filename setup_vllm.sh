#!/bin/bash
python3 -m venv venv_vllm
source venv_vllm/bin/activate
export VLLM_VERSION=0.5.4
pip3 install -r requirements.txt
pip3 install "transformers==4.43.2"
pip3 install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-${VLLM_VERSION}-cp38-abi3-manylinux1_x86_64.whl
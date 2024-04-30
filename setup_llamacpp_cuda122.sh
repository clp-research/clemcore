#!/bin/bash
python3 -m venv venv_llamacpp
source venv_llamacpp/bin/activate
pip3 install -r requirements.txt
# install using pre-built wheel with CUDA 12.2 support:
pip3 install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
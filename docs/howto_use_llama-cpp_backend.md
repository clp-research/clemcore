# Setup and usage of llama.cpp clembench backend
This guide covers the installation and usage of the llama.cpp-based backend for clembench. This backend allows the use 
of models in the GGUF format, supporting pre-quantized model versions and merged models. The setup varies by available 
hardware backend and operating system, and models may need to be loaded with specific arguments depending on the setup.  
## Content
[Setup](#setup)  
[Model loading](#model-loading)
## Setup
The clembench llama.cpp backend relies on the llama-cpp-python library, which wraps C++ llama.cpp. To allow the usage of 
specific hardware, specially GPUs, the installation must include a fitting version of llama.cpp. This may entail 
compiling llama.cpp, but pre-compiled versions for specific hardware are available.  
Since this is specific to the available hardware, please refer to the [llama-cpp-python installation instructions](https://llama-cpp-python.readthedocs.io/en/latest/#installation) 
to install the library. It is recommended to use one of the pre-built wheels for the available hardware, as this does not require a C++ compiler 
and compiling llama.cpp during the installation.
### Sample setup script
The following example shell script installs the clembench llama.cpp backend with support for CUDA 12.2 GPUs:
```shell
# create separate venv for running the llama.cpp backend:
python3 -m venv venv_llamacpp
source venv_llamacpp/bin/activate
# install basic clembench requirements:
pip3 install -r requirements.txt
# install llama-cpp-python using pre-built wheel with CUDA 12.2 support:
pip3 install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
```
## Model loading
The clembench llama.cpp backend downloads model files from HuggingFace model repositories. See the [model registry readme](model_backend_registry_readme.md).  
By default, the clembench llama.cpp backend loads all model layers onto the available GPU(s). This requires that during 
setup, proper llama.cpp GPU support fitting the system hardware was installed.  
Optionally, models can be loaded to run on CPU (using RAM instead of GPU VRAM). This is required if llama-cpp-python was 
installed without GPU support. This can be done by passing a JSON object to the clembench CLI scripts, or a Python `dict` 
to the model loading function of the clembench `backends`.  
The JSON object/`dict` has to contain the model name as defined in the [model registry](model_backend_registry_readme.md) 
and the key `execute_on` with string value `gpu` or `cpu`:
```python
model_on_gpu = {'model_name': "openchat_3.5-GGUF-q5", 'execute_on': "gpu"}
model_on_cpu = {'model_name': "openchat_3.5-GGUF-q5", 'execute_on': "cpu"}
```
For clembench CLI scripts, the JSON object is given as a "-delimited string:
```shell
# run the taboo clemgame with openchat_3.5-GGUF-q5 on CPU:
python3 scripts/cli.py run -g taboo -m "{'model_name': 'openchat_3.5-GGUF-q5', 'execute_on': 'cpu'}"
```
Alternatively, the number of model layers to offload to GPU can be set by using the `gpu_layers_offloaded` key with an 
integer value:
```python
model_15_layers_on_gpu = {'model_name': "openchat_3.5-GGUF-q5", 'gpu_layers_offloaded': 15}
```
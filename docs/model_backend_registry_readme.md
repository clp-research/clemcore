# Model Registry
The model registry holds information and settings for each of the models currently supported by clembench.  
It is stored as a JSON file in the `backends` directory. This file contains a list of model entry objects.  
## Model Entry Components
Each object in the list contains two mandatory key/values:  
`model_name`(string): The name the model is identified by in clembench. This is also the specific version name of the model to be used by the backends. (*Might change in future versions.*)  
`backend`(string): The name of the backend that handles this model.  
Further key/values depend on the backend handling the model.  
### Local Huggingface Backend
This backend requires these **mandatory** key/values:  
`huggingface_id`(string): The full huggingface model ID; huggingface user name / model name. Example: `01-ai/Yi-34B-Chat`  
`premade_chat_template`(bool): If `true`, the chat template that is applied for generation is loaded from the model repository on huggingface. If `false`, the value of `custom_chat_template` will be used if defined, otherwise a generic chat template is applied (highly discouraged).  
`eos_to_cull`(string): This is the string representation of the model's EOS token. It needs to be removed by the backend to assure proper processing by clembench. Example: `<|im_end|>` (This is mandatory as there are models that do not define this in their tokenizer configuration.)  

The following key/values are **optional**, but should be defined for models that require them for proper functioning:  
`requires_api_key`(bool): If `true`, the backend will load a huggingface api access key/token from `key.json`, which is required to access 'gated' models like Meta's Llama2.  
`custom_chat_template`(string): A jinja2 template string of the chat template to be applied for this model. This should be set if `premade_chat_template` is `false` for the model, as the generic fallback chat template that will be used if this is not defined is likely to lead to bad model performance.  
`slow_tokenizer`(bool): If `true`, the backend will load the model's tokenizer with `use_fast=False`. Some models require the use of a 'slow' tokenizer class to assure proper tokenization.  
`output_split_prefix`(string): The model's raw output will be rsplit using this string, and the remaining output following this string will be considered the model output. This is necessary for some models that decode tokens differently than they encode them, to assure that the prompt is properly removed from model responses. Example: `assistant\n`
# Backend Classes
Model registry entries are mainly used for two classes: `backends.ModelSpec` and `backends.Model`.
## ModelSpec
The `backends.ModelSpec` class is used to hold model-specific data, as defined in a model entry, and default 
generation parameters. All backend functions and methods expect instances of this class as arguments for model loading.  
As part of a benchmark run, `ModelSpec` is initialized using the model name only, and the settings are loaded from the 
model registry, from the first entry with the given name, unifying with the entry contents.  
For testing and prototyping, a `ModelSpec` can be initialized from a `dict` with the same structure as a model entry, 
using `ModelSpec.from_dict()`.
## Model
The `backends.Model` class is used for fully loaded model instances ready for generation.
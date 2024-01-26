# How to add a model to the HuggingFace backend
This HowTo explains how to add a LLM hosted on the HuggingFace (HF) model repository to the local HuggingFace backend of clembench.
## HuggingFace models
Each model hosted on [HuggingFace](http://huggingface.co) is identified by its **model ID**, which is the combination of
the **model uploader's username** and the individual **model name**.  
**For example:** For the OpenChat 3.5 model, the model ID is `openchat/openchat_3.5`, as `openchat` is the uploader's user name and 
`openchat_3.5` is the model name.  
This model ID is all that is needed to access *ungated* models hosted on HuggingFace.  
Accessing *gated* models, like Meta's Llama2, requires an HF API access key/token. HF API tokens are acquired via your 
user profile on the HF website. Make sure that the HF account used to acquire the access key has been granted access to 
the gated model you want to add. This API key needs to be added to `key.json` in the clembench root directory to be available for loading gated model data.
## Workflow
### I. Check the model card on HuggingFace
You should thoroughly read the model card for the model to be added to be informed about individual aspects. It's also a
good idea to look at the community tab of a model repository to see if there are common issues with the model.
### II. Check the model's tokenizer
The clembench HuggingFace local backend relies on the `transformers` and indirectly on the `tokenizers` libraries for 
model-dependent input tokenization. It also relies the **chat template** utility of the libraries' tokenizer classes. 
This first step is to make sure that a candidate model hosted on HuggingFace has the required configuration to be used 
with the clembench backend.  
To perform a preliminary check for compatibility, run `python3 backends/initial_hf_check.py -m <MODEL ID>`.  
**For example:** `python3 backends/initial_hf_check.py -m openchat/openchat_3.5` to check the OpenChat 3.5 model.  
The `initial_hf_check.py` script will show the applied template and warn about common issues, but does not cover all 
edge cases. It also takes the flags `-i` to show the tokenizer's information and `-t` to show the configured chat 
template in jinja string format, which can be useful for modification into a custom template for the model.  
The initial check script applies the same preprocessing as the backend.  
### III. Add the model's information to the local HuggingFace model registry
Open `backends/hf_local_models.json` in your editor of choice. This file contains entries for all models supported by 
the huggingface-local backend. To make a new model available, an entry for it needs to be added to this registry.  
#### Basic model entry
A minimal model entry contains the model name, the backend to handle it, its HF ID, a bool that determines if a premade 
chat template for it will be loaded from HF and the EOS string to be culled from its outputs:  
```
{
  "model_name": "Mistral-7B-Instruct-v0.1",
  "backend": "huggingface",
  "huggingface_id": "mistralai/Mistral-7B-Instruct-v0.1",
  "premade_chat_template": true,
  "eos_to_cull": "</s>"
}
```
#### Chat template
If the model to be added passed the initial check without any issue, use `"premade_chat_template": true` in its registry 
entry. This indicates that the model's tokenizer properly applies a chat template that works without any further editing.  
If it does not pass the check or otherwise requires chat template changes, the entry must contain 
`"premade_chat_template": false` and include the custom chat template to be used in jinja2 string format.  
**For example:**  
```
{
  "model_name": "sheep-duck-llama-2-70b-v1.1",
  "backend": "huggingface",
  "huggingface_id": "Riiid/sheep-duck-llama-2-70b-v1.1",
  "premade_chat_template": false,
  "custom_chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### User:\\n' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'system' %}{{ '### System:\\n' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ '### Assistant:\\n' + message['content'] + '\\n\\n' }}{% endif %}{% if loop.last %}{{ '### Assistant:\\n' }}{% endif %}{% endfor %}",
  "eos_to_cull": "</s>"
}
```
#### Slow tokenizer handling
If the model requires the use of the 'slow' tokenizer class, which should be noted on the model card, the model entry 
must contain `"slow_tokenizer": true`.  
**For example:**  
```
{
  "model_name": "SUS-Chat-34B",
  "backend": "huggingface",
  "huggingface_id": "SUSTech/SUS-Chat-34B",
  "premade_chat_template": false,
  "custom_chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### Human: ' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ '### Assistant: ' + message['content'] }}{% endif %}{% if loop.last %}{{ '### Assistant: ' }}{% endif %}{% endfor %}",
  "slow_tokenizer": true,
  "eos_to_cull": "<|endoftext|>"
}
```
#### Output split string
The model to be added might use an uncommon tokenizer, which can lead to discrepancies between prompt and decoded model 
output, requiring the model output to be split to be properly handled by clembench. In this case, the string that 
predeces the model output proper needs to be contained in the model entry. (This will likely be found in testing the 
model.)  
**For example:**  
```
{
  "model_name": "Yi-34B-Chat",
  "backend": "huggingface",
  "huggingface_id": "01-ai/Yi-34B-Chat",
  "premade_chat_template": false,
  "custom_chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = true %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
  "slow_tokenizer": true,
  "output_split_prefix": "assistant\n",
  "eos_to_cull": "<|im_end|>"
}
```
#### HF API key requirement
If the model to be added is *gated*, the model entry must contain `"requires_api_key": true`. Make sure that `key.json` 
exists and has a viable HF API access key when the model is to be used.  
**For example:**  
```
{
  "model_name": "llama-2-7b-hf",
  "backend": "huggingface",
  "requires_api_key": true,
  "huggingface_id": "meta-llama/llama-2-7b-hf",
  "premade_chat_template": true,
  "eos_to_cull": "</s>"
}
```
#### Further model registry information
See [the model registry readme](model_backend_registry_readme.md) for more information on the model registry.   
### IV. Test the model
#### Run HelloGame
Run clembench with the `hellogame` clemgame. See the corresponding documentation for HowTo.  
This produces interactions and requests files in JSON format in the `results/hellogame` directory. Specific files can be 
found in `results/hellogame/records/<MODEL NAME>/0_greet_en/` episode subdirectories.
#### Check requests files
The requests file of each episode contains the prompts given to the model and its outputs.  
Check the `modified_prompt_object` values for proper application of the chat template.  
Then check if there *is* generated text and if the model outputs match the `modified_prompt_object` before the generated 
text.  
Finally, check if the model output ends with a EOS string. This string needs to be culled, as noted above, and proper 
culling is checked in the next step.
#### Check interactions files
The interactions files contain processed outputs in the form they are relevant to clembench.  
Model replies in the interaction files should not contain any model-specific EOS token strings.  
Check if the model replies end in an EOS string. If they do, add this exact string to the EOS culling in the backend 
code as shown above.
#### Repeat after changes
If you made any changes to the code after the first test, run the test again and check the files to make sure that they 
now have proper contents.
### V. Share your code
If you have successfully run the tests above, open a pull request for the clembench repository.  
You can also run the benchmark with your added model if you have the necessary hardware available - if you do, please 
share the results by contributing them to the clembench-runs repository.
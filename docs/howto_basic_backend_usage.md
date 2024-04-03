# How to use the clembench model backend directly
This guide covers using the clembench Model backend class outside of full clemgames.
## Requirements
clembench needs to be fully set up, including all requirements of the models to be used, like the HuggingFace 
`transformers`-based backend requirements listed in `requirements_hf.txt`. See [the basic howto](howto_run_benchmark.md) 
for more information, specially for the handling of API keys.
## The backends.Model class
All remote API requests and generation requests to locally run models are handled by child classes of the `backends.Model` 
class.  
All `Model` child classes implement the `generate_response()` method. `generate_response()` expects a `List[Dict]` 
messages object containing an exchange of chat messages. Each message contains a role and text content. The method 
returns the full prompt, full output and generated text as a tuple of strings. See [the model addition howto](howto_add_models.md) 
for more information.
## Basic example
Load a supported model and generate a reply:
```python
import backends

model_name = "zephyr-7b-beta"

model = backends.get_model_for(model_name)

messages = [
    {'role': "user", 'content': "Hello!"},
    {'role': "assistant", 'content': "Hello! How can I help you?"},
    {'role': "user", 'content': "Tell me the name of the capital of Australia."},
]

prompt, response, response_text = model.generate_response(messages)

print(f"{model_name} reply:")
print(response_text)
```
## Multiple models example
Loop over a list of supported model names and generate a reply to the same messages with each:
```python
import backends

model_names = ["zephyr-7b-alpha", "zephyr-7b-beta", "openchat-3.5-0106"]

messages = [
    {'role': "user", 'content': "Hello!"},
    {'role': "assistant", 'content': "Hello! How can I help you?"},
    {'role': "user", 'content': "Tell me the name of the capital of Australia."},
]

for model_name in model_names:
    model = backends.get_model_for(model_name)
    
    prompt, response, response_text = model.generate_response(messages)

    print(f"{model_name} reply:")
    print(response_text)
    
    # remove Model instance to free up memory:
    del model
```
The removal of `Model` instances may be omitted if enough memory (VRAM) is available to load multiple models at the same 
time. clembench **does not** handle memory demands and limitations, and loading models when remaining memory space is 
insufficient will lead to `torch`/CUDA crashes.
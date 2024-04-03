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

model_names = ["zephyr-7b-alpha", "zephyr-7b-beta", "openchat-3.5-0106", "gpt-3.5-turbo-0125"]

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
The removal of `Model` instances may be omitted if enough memory (VRAM) is available to load multiple models locally at 
the same time. clembench **does not** handle memory demands and limitations, and loading models when remaining memory space is 
insufficient will lead to `torch`/CUDA crashes.  
As remote models accessed via API (like "gpt-3.5-turbo-0125" above) do not require local memory, multiple `Model` 
instances using these models can be active at the same time without issues.
## clembench backends details
While the clembench backends can simply be used as shown above, certain implementation details may be helpful.
### Supported Models & Model Registry
A model is supported by clembench if it has an entry in the [model registry](../backends/model_registry.json). Model 
entries contain all necessary information to load the model and use it to generate chat replies. The model registry 
supplied on the clembench repository covers tested and working models. See the 
[model registry documentation](model_backend_registry_readme.md) for more information.
### ModelSpec class
The `backends.ModelSpec` class holds information retrieved from the model registry. In addition, it can store sampling 
parameters (like temperature). Note: Non-standard sampling parameters must be accepted by specific model backends.
### Model chat functionality
While most remote APIs apply chat formatting server-side, local clembench backends apply model-appropriate chat 
templates for generation. This assures that the input follows the multi-turn format the model was trained on. The 
first tuple element returned by `generate_response()` for the latter contains the applied formatting.
### backends.get_model_for()
The `backends.get_model_for()` takes either a model name, as defined in a model registry entry, a `dict` containing the 
necessary model information or a `backends.ModelSpec` instance.  
When only a model name is passed (as in the examples above), the first model entry matching the name is retrieved to 
instantiate a `ModelSpec`, which is used to load the corresponding model.  
Passing a `dict` allows to select a specific model version. This is useful for models that have multiple registry 
entries: For example "openchat-3.5", which has a remote 'openai-compatible' entry and an entry for running it on the 
local HuggingFace backend. Using a `dict` also allows to instantiate a model with non-standard sampling parameters 
directly.  
See the [backends code](../backends/__init__.py) for the `ModelSpec` class and intermediate uses of it.
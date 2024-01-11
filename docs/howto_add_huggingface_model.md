# How to add a model to the HuggingFace backend
This HowTo explains how to add an *ungated* LLM hosted on the HuggingFace (HF) model repository to the local HuggingFace 
backend of clembench.
## HuggingFace models
Each model hosted on [HuggingFace](http://huggingface.co) is identified by its **model ID**, which is the combination of
the **model uploader's username** and the individual **model name**.  
**For example:** For the OpenChat 3.5 model, the model ID is `openchat/openchat_3.5`, as `openchat` is the uploader's user name and 
`openchat_3.5` is the model name.  
This model ID is all you need to access *ungated* models hosted on HuggingFace.
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
### III. Add the model's information to the local HuggingFace backend code
*Note: This part is likely to change in the future.*  
Open `backends/huggingface_local_api.py` in your editor of choice.  
#### Model constant
Below the imports and logger assignment, add a **model constant** for the model you are adding. Model constants are 
all-capital, and their value is the model name as a string (only the name, not the full HF model ID).  
**For example:** `MODEL_OPENCHAT_3_5 = "openchat_3.5"`  
#### Supported models
Add the new model constant to the `SUPPORTED_MODELS` list.  
This is required to make the model available to be used with clembench.
#### Chat template setup
If the model to be added passed the initial check without any issue, its model constant can be added to the 
`PREMADE_CHAT_TEMPLATE` list. This indicates that the model's tokenizer properly applies a chat template that works 
without any further editing.  
Below `PREMADE_CHAT_TEMPLATE`, custom chat templates are assigned for models that need further edits to the template for
it to work with clembench. (More on this TBD)
#### Slow tokenizer handling
If the model requires the use of the 'slow' tokenizer class, which should be noted on the model card, add the model 
constant to the `SLOW_TOKENIZER` list.
#### Model loading
In the `HuggingfaceLocal` class method `load_model`, add model uploader username handling.  
**For example:**  
```
elif model_name in [MODEL_OPENCHAT_3_5]:  # openchat models
            hf_user_prefix = "openchat/"
```
If the model uploader of the model to be added is already handled, simply add the model constant to the list in the 
corresponding el/if clause.
#### Chat template application
If you have added a custom template, it needs to be applied to the tokenizer.  
**For example:**  
```
# apply proper chat template:
if model_name not in PREMADE_CHAT_TEMPLATE:
    elif model_name in OPENCHAT:
        self.tokenizer.chat_template = openchat_template
```
#### EOS culling
If the model ends outputs with a specific token, add removal of the corresponding string to the `generate_response` 
method.  
**For example:**  
```
# remove openchat EOS token at the end of output:
if response_text[-15:len(response_text)] == "<|end_of_turn|>":
    response_text = response_text[:-15]
```
It might not be obvious or noted on the model card if the model to be added does this, but the step in the workflow 
checks for this.
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
# Huggingface Prototyping Check Methods
The huggingface-local backend offers two methods to check messages lists that clemgames might pass to the backend 
without the need to load the full model weights. This allows to prototype clemgames locally with minimal hardware demand
and prevent common issues.
## Messages Checking
The `check_messages` method of the huggingface-local backend class takes a `messages` list and a `model: str` name as 
arguments.  
It will print all anticipated issues with the passed messages list to console if they occur. It also applies the given 
model's chat template to the messages as a direct check. It returns `False` if the chat template does not accept the 
messages and prints the outcome to console.
## Context Limit Checking
The `check_context_limit` method of the huggingface-local backend class takes a `messages` list and a `model: str` name 
as required arguments. Further arguments are the number of tokens to generate `max_new_tokens: int` (default: `100`), 
`clean_messages: bool` (default: `False`) to apply message cleaning as the generation method will, and `verbose: bool` 
(default: `True`) for console printing of the values.  
It will print the token count for the passed messages after chat template application, the remaining number of tokens
(negative if context limit is exceeded) and the maximum number of tokens the model allows as generation input.  
The method returns a tuple with four elements:  
- `bool`: `True` if context limit was not exceeded, `False` if it was.
- `int`: number of tokens for the passed messages.
- `int`: number of tokens left in context limit.
- `int`: context token limit.  
## Usage Examples
Run `huggingface_local_api.py` for example outputs.  
Example code is included in `backends/huggingface_local_api.py`.
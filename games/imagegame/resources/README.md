# Imagegame resources

## Adding languages
* get translations of the files in `imagegame/resources/initial_prompts/to_translate/`
* create a language folder under `imagegame/resources/initial_prompts/` and save the translated prompts as 
  * `player_a|b_prompt_header.template` and
  * `prompt_question.template`  
  * (removing the language prefix)
* create a new entry in `imagegame/resources/localization_utils.py` from the translations in `responses.template` (make sure to include the colon in the language specific version)
* machine translations: treat them as separate languages by using a name such as `de_google`  
  * also see [google-translations.md](initial_prompts/google-translations.md)
* run `imagegame/instancegenerator.py` as usual to create the instances in `imagegame/in/`
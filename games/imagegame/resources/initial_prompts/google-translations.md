This file contains information about how the Google translations have been created.

**Translations from:** https://translate.google.de/  
**Created at:** 12-Jun-2024  
**Updated at:** 23-Jul-2024 (`command` instead of `instruction`)  
**Languages:** de, es, ru, te, tk, tr

### Adjustments made:

The adjustments are necessary for successful parsing and to ensure comparability.

1. The tag `command:` must be identical (case-insensitive) at:
   - player_a_prompt_header: 3x in the example (l.22, 25, 28)
   - localization_utils

1. The terminate token `DONE` must be
   - translated into respective language
   - identical at:
     - player_a_prompt_header: 1x at end of the longest paragraph (l.17), 1x in the example (l.28)
     - localization_utils
   - if there are different translations: try out what [DeepL](https://www.deepl.com/de/translator) says.

1. The grids must be identical to those in `to_translate/`:
   - 2x player_a_prompt_header
   - 2x player_b_prompt_header

1. `GRID_DIMENSION` should not be translated.
   - for the translation replace `GRID_DIMENSION` by `5`, afterwards replace `5` by `GRID_DIMENSION` again
   - player_a_prompt_header (l.30)
   - player_b_prompt_header (l.19)

1. Line number should be same as in corresponding files in `to_translate/` (no extra line breaks/paragraphs)
   - player_a_prompt_header
   - player_b_prompt_header
   - prompt_question

1. Prompt Question is extracted from line 17, player_a_prompt_header
   - prompt_question

1. words in response patterns don't have to be lowered because re.IGNORECASE is used. Don't use python `str.lower()` function because in some languages it can lead to unexpected results.
   - localization_utils


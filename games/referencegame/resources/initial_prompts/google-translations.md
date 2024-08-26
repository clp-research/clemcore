This file contains information about how the Google translations have been created.

**Translations from:** https://translate.google.de/  
**Created at:** 12-Jun-2024  
**Languages:** de, es, ru, te, tk, tr

### Adjustments made:

The adjustments are necessary for successful parsing and to ensure comparability.

1. Tag `expression:`
   - must be translated into respective language
   - must be identical (case-insensitive) at:
     - player_a_prompt_header: 1x in line 19
     - localization_utils

1. Tag `answer:`
   - must be translated into respective language
   - must be identical (case-insensitive) at:
     - player_b_prompt_header: 1x in line 21
     - localization_utils

1. first, second, third must be identical (case-insensitive) at:
   - player_b_prompt_header:
     - line 5: "first, second, or third"
     - line 7, 11, 15
   - localization_utils
   - If there are different translations: try out what [DeepL](https://www.deepl.com/de/translator) says. If that doesn't help: Use the translations from line 7, 11 and 15 replacing the ones in line 5

1. don't translate capitalised words
   - player_a_prompt_header: line 8, 12, 16 (`TARGET_GRID`, `SECOND_GRID`, `THIRD_GRID`)
   - player_b_prompt_header: line 9, 13, 17, 19 (`FIRST_GRID`, `SECOND_GRID`, `THIRD_GRID`, `TARGET_EXPRESSION`)

1. line number should be identical to files in `/to_translate` (no extra line breaks)
   - player_a_prompt_header
   - player_b_prompt_header

1. words in response patterns don't have to be lowered because re.IGNORECASE is used. Don't use python `str.lower()` function because in some languages it can lead to unexpected results.
   - localization_utils

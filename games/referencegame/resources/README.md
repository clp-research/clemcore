# Referencegame resources

## Creating instances
* run `create_grids.py` (change `GRID_FILE` and parameters in `create_random_grids` to generate new instances, the other categories are hard-coded)
* to add new languages: 
  * get translations of the three files in `referencegame/resources/initial_prompts/to_translate/`
  * create a language folder under `referencegame/resources/initial_prompts/` and save the translated prompts as `player_a|b_prompt_header.template` (removing the language prefix)
  * create a new entry in `referencegame/resources/localization_utils.py` from the translations in `responses.template` (make sure to include the colon in the language specific version)
* run `referencegame/instancegenerator.py` to create the instances in `referencegame/in/`


## Different versions
Grids consist of 5x5 matrices filled with '▢'s and 'X's to form low-level image representations like this:
```
X X X X X
X ▢ ▢ ▢ X
X ▢ ▢ ▢ X
X ▢ ▢ ▢ X
X X X X X
```

### grids_v0.9.json
* contains `easy_grids` and `hard_grids`
* (probably manually created)

### grids_v1.0.json
* contains only `hard_grids`
* (seem to be different from v01)
* used in version 1.0 of referencegame to create distorted distractor images (by randomly removing 2 or 4 Xs)

### grids_v1.5.json
* contains 
  * `line_grids_rows` and `line_grids_columns` which contain all possibilities of one and four lines of Xs
  * `diagonal_grids` which combine diagonal grids in v01 and other possible diagonal shapes
  * `shape_grids`, which are selected from the grids in v01 and v02
  * `letter_grids`, which represent alphabetic symbols
  * `random_grids`, which contains 10 grids with 10 randomly distributed Xs

* used in version 1.5 to create triplets by selecting each grid as target once and selecting two distractors each from within the category, based on the smallest edit distance to the target

### initial_prompts changes in version 1.5
Player A: 
* Changed expression and wording trying to force more diverse referring expressions
* Removed examples in order to avoid biasing the models and changed the wording to still accomplish instruction following

Player B: 
* Changed expression and wording trying to force more attention to referring expressions
* Removed examples in order to avoid biasing the models and changed the wording to still accomplish instruction following

### Conceptual changes in version 1.5
To balance the distribution of labels for Player 2, three instances are generated from each grid triplet, where the target is in one of the three possible positions each. This allows to later calculate the score with respect to outperforming random guessing.

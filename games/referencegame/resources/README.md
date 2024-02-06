## Different versions
Grids consist of 5x5 matrices filled with '▢'s and 'X's to form low-level image representations like this:
```
X X X X X
X ▢ ▢ ▢ X
X ▢ ▢ ▢ X
X ▢ ▢ ▢ X
X X X X X
```

###grids_v01.json
* contains ```easy_grids``` and ```hard_grids```
* (probably manually created)

### grids_v02.json
* contains only ```hard_grids```
* (seem to be different from v01)
* used in version 1.0 of referencegame to create distorted distractor images (by randomly removing 2 or 4 Xs)

### grids_v03.json
* basis for less biased version 2.0 of referencegame
* contains 
  * ```line_grids_rows``` and ```line_grids_columns``` which contain all possibilities of one and four lines of Xs
  * ```line_grids_diagonal``` which are selected from the grids in v01
  * ```shapes```, which are selected from the grids in v01 and v02 (de-duplicated) 
  * ```letters```, which represent alphabetic symbols
    * two ideas on how to create instances from those:
        1. manually create representation triplets for supposedly hard combinations (state so far, but option 2 would be more desirable)
        2. manually create representations for all letters and combine based on edit distance to create harder and easier versions
  
### initial_prompts changes in version 2.0
#### Player A: 
* Changed expression from "Filled with T." to "Looks like a T"
#### Player B: 
* Changed expression from "Filled with T." to "Looks like a T"
* Removed second example (that implicitly guided the model to select the grid with the most Xs)
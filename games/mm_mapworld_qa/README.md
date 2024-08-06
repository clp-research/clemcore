## Map World Game - Exhaustive Exploration with Question Answering (EE-qa)

Implemented by: Yan Weiser

### How to play

Instances to play are already provided in this repository. In order to run them use the following command while being in the root clembench directory:

```
python3 scripts/cli.py run -g mm_mapworld_qa -m [model_to_run]
```
after running the game you can create transcripts and scores for each instance by running 
```
python3 scripts/cli.py transcribe -g mm_mapworld_qa
```
and 
```
python3 scripts/cli.py score -g mm_mapworld_qa
```
Scoring will also create animations (gifs) of the payers movement for each instance.

### Requirements

This Game can only be played by multimodal models. This means that the `supports_images` tag in the `clembench/backends/model_registry.json` file needs to be true for that model.

### Creating new instances

While instances are already provided, you might want to alter them or create more than the 50 that come with the repository. To do that you can make your changes to the `clembech/games/mm_mapworld_qa/instancegenerator.py` file and then run this file directly. All images and room categories are provided with the repository.

Here are some changes that can be done easily:
- alter the number of rooms for the different graph sizes by changing the values in the `SIZES` dictionary
- change the grid sizes on which the maps are created by setting the (X,Y) tuples in the `GRIDS` variable
- create more or less instances for each experiment by setting the desired number as the `NUM_INSTANCES` variable
- change the amount of ambiguity in a certain difficulty by adjusting the values in the `AMBIGUITIES` variable

### Scores

Besides the Main score that is being used for the final clemscore, other scores are also calculated.

- **Aborted** - is a binary score, indicating whether the instance was aborted at some point during the run
- **Success** - is a binary score that is 1 if all rooms have been visited and the game was not aborted
- **Lose** - is the inverse of **Success**
- **moves** - the number of moves taken by the player
- **valid_moves** - the number of valid moves taken by the player. A move is valid if there is a room in the chosen direction.
- **invalid_moves** - is difference between **moves** and **valid_moves**
- **seen** - the number of rooms seen by the player. A room has been seen if the player was in a room adjacent to it.
- **visited** - the number of rooms visited by the player
- **question_answering** - the fraction of correctly answered questions out of all answered questions
# Codenames

Implemented by: Lara Pfennigschmidt (lpfennigschmidt)

[Codenames](https://codenames.game) is a multi-player game with two teams (red and blue) playing against each other. Each team consists of a Cluegiver and one or multiple Guessers. The Cluegiver has to give a single word that relates to a number of words on a 5x5 word grid that were assigned to their team, to make the team's guesses uncover their teams words. Clue words cannot be morphologically related to other words on the board and have to be related to the board words' meanings, not to any syntactic properties. The number of words the clue word relates to is also given to the Guessers and constrains the number of guesses they can make. After each guess the correctness is evaluated: if the guess was indeed a team word (whether initially targeted by the Cluegiver or not) the Guessers can make another guess if it is still within the number of targets. If the guess was incorrect (either an opposing team word or an 'innocent' word) the team's turn ends there and the opponent team gets their turn. If the guess uncovered an assassin word, the team loses immediately. A team wins when their last team word gets uncovered, it does not matter which team uncovers it, or if the other team gets assassinated.

The game tests creative abilities to come up with related words that are close enough to target words but far enough from other team's words and especially from the assassin word. Players have to make ad-hoc word associations and constrain them according to other words on the board. Guessers have to rank words on the board by semantic closeness and guess closest words first.

The implemented version only deploys one Cluegiver and one Guesser on the same team, the other team is mocked with an ideal behaviour, uncovering n of their own team words each turn, with n corresponding to the opponent difficulty. The Cluegiver is asked for a clue word and a list of targets from which the number of targets is inferred. The list of targets is also only needed to calculate more specific target-related scores. The Guesser is asked for a list of guesses best relating to the clue word.

The implementation of the GameMaster can be found in [master.py](master.py), the Cluegiver and Guesser implementations as well as the mock players can be found in [players.py](players.py). The game state is saved in the CodenamesBoard class which is implemented in [board.py](board.py). The scorer is implemented in [scorer.py](scorer.py).

## Experiments
The set of [experiments](resources/experiments.json) contains word-list related experiments scrutinising model performance on boards with differing word frequency, word ambiguity, and word concreteness. Additional experiments are concerned with game mechanics such as risk (having more assassins), and mock opponent difficulty. A last experiment plays around with the word assignment mechanism, assigning closely related words to the same group in one condition and randomly to all groups in the other condition. (Experiments on the original Codenames words and different board sizes are currently excluded from the experiment set but their configuration can be found in [resources/all_experiments.json](resources/all_experiments.json)).

## Prompts
Can be found in [initial prompts](resources/initial_prompts/) and [intermittent prompts](resources/intermittent_prompts/) respectively.

## Error Handling and Mitigation
Because of typical errors like hallucinations or not following the correct format and due to having a multi-turn game setup I implemented some error-mitigating behvaiour that is active in the generous instances. These can be toggled via flags already set in the instances. The flags are:

- `STRIP WORDS`: strips clue words, targets, and guesses of unnecessary punctuation characters, active in both strict and generous mode
- `REPROMPT ON ERROR`: whether models are reprompted on non-mitigatable errors and given feedback about the nature of their errors
- `IGNORE RAMBLING`: removes all additional text after it encounters two newlines
- `IGNORE FALSE TARGETS OR GUESSES`: removes hallucinated targets or guesses or targets/guesses that were available in a previous turn but are not anymore and would therefore be invalid
- `IGNORE NUMBER OF TARGETS`: previously needed to avoid the error of models giving only the number of targets instead of the list of targets, but only occurred because of an error in the turn history which was then fixed, so this flag is actually not needed anymore.

The implementation of all error types and the respective reprompt feedback messages can be found in [validation_errors.py](validation_errors.py).
Most flags are only active in generous mode, the only flag active in both modes is `STRIP WORDS`.

# Instance Generation

Instances for all experiments defined in [resources/experiments.json](resources/experiments.json) can be generated with:

```bash
python3 games/codenames/instancegenerator.py [-v variable_name] [-e experiment_name] [--keep] [--strict]
```

Currently, the instance generation is reproducible, thus generating completely new instances requires the random seed to be changed in [constants.py](constants.py). The generator can also generate only specific instances for single experiments or variables with -e and -v respectively. To keep all other instances and only re-generate specific ones, additionally make use of the --keep flag.

The used wordlists to generate instances can be found in [resources/cleaned_wordlists](resources/cleaned_wordlists). To extend or create new wordlists, please edit or add them to [resources/wordlists/](resources/wordlists/) and run the wordlist_cleaner.py to clean them and automatically put the cleaned versions into [resources/cleaned_wordlists](resources/cleaned_wordlists).
For further information on the creation of the already existing wordlists, please refer to my [other repository](https://github.com/lpfennigschmidt/thesis-codenames/tree/main/board%20generation).

The set of experiments in [resources/experiments.json](resources/experiments.json) is reduced to the most important. For a config of all experiments run for the thesis, please consult [resources/all_experiments.json](resources/all_experiments.json).

# Run

## Preparation
```bash
source prepare_path.sh
```

## Running the game
This will run all experiments that have generated instances in [in/instances.json](in/instances.json). These are the generous instances with all error-mitigating flags set to True. To run the strict instances from [in/strict_instances.json](in/strict_instances.json) (where only the character stripping is in place), use the -i flag in the cli (see fourth command below).
The following commands show a set of possibilities to run the game with different model players.

```bash
python3 scripts/cli.py run -g codenames -m fsc-openchat-3.5-0106

python3 scripts/cli.py run -g codenames -m mock

python3 scripts/cli.py run -g codenames -m fsc-openchat-3.5-0106 human

python3 /cli.py run -g codenames -m fsc-openchat-3.5-0106 [-i strict_instances -r ./strict_results]
```

The behaviour of the mock player can be switched between ideal and random mock player by setting the flag `MOCK_IS_RANDOM` in [players.py](players.py) to `False` or `True` respectively.

## Transcribing

To transcribe the model interactions into transcripts (html and tex):

```bash
python3 scripts/cli.py transcribe -g codenames
```

## Scoring
To score previously run instances, run:

```bash
python3 scripts/cli.py score -g codenames
```

Episode-level scores

- Sensitivity (Recall) and Specificity (Episode Negative Recall)
- Efficiency (words revealed/number of turns / 2)
- Quality Score: harmonic mean of Sensitivity and Efficiency
- Won, Lost, Aborted, Lost through assassin, number of turns
- valid and invalid requests, request success ratio
- average turn-level scores

Turn-level scores

- number of targets, team word precision, recall, and F1 for the Cluegiver
- number of guesses, team word precision, recall, and F1 for the Guesser, as well as target precision, recall, and F1

## Evaluation
To create evaluation tables, run the following that apply:

```bash
python3 evaluation/codenames_eval.py [-m all|models|experiments|errors|clemscores] [-r results_path]
python3 evaluation/codenames_differences.py
python3 evaluation/latex_table_generator.py
```

The latex table generator requires the latextable package.


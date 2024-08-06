# Codenames

The implementation of the GameMaster can be found in [master.py](master.py), the Cluegiver and Guesser implementations as well as the mock players can be found in [players.py](players.py). The game state is saved in the CodenamesBoard class which is implemented in [board.py](board.py). The scorer is implemented in [scorer.py](scorer.py).

## Prompts
Can be found in [initial prompts](resources/initial_prompts/) and [intermittent prompts](resources/intermittent_prompts/) respectively.

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

## Evaluation
To create evaluation tables, run the following that apply:


```bash
python3 evaluation/codenames_eval.py [-m all|models|experiments|errors|clemscores] [-r results_path]
python3 evaluation/codenames_differences.py
python3 evaluation/latex_table_generator.py
```

The latex table generator requires the latextable package.


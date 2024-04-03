# Running the benchmark

## Setting up

### Install dependencies

```
pip install -r requirements.txt
```

### API Key

Create a file `key.json` in the project root and paste in your api key (and organisation optionally).

```
{
  "openai": {
            "organisation": "<value>", 
            "api_key": "<value>"
            },
  "anthropic": {
            "api_key": "<value>"
            },
  "alephalpha": {
            "api_key": "<value>"
            }
}
```

Note: You can look up your api key for OpenAI at https://platform.openai.com/account/api-keys and for Anthoropic
at https://console.anthropic.com/account/keys, AlephAlpha can be found
here: https://docs.aleph-alpha.com/docs/introduction/luminous/

### Supported models

Supported models are listed in the [model registry](../backends/model_registry.json).  
To use a supported model as shown below, use the model registry entry's `model_name`.  
See the [model registry documentation](model_backend_registry_readme.md) for more information.

## Validating your installation

Add keys to the API providers as described above.

Go into the project root and prepare path to run from cmdline

```
source prepare_path.sh
```

Then run the cli script

```
python3 scripts/cli.py run -g taboo -m gpt-3.5-turbo
```

The `-m` option tells the script which model to use. Since taboo is a two player game, we could theoretically also 
let two different models play the game which would look like:

```
python3 scripts/cli.py run -g taboo -m gpt-3.5-turbo-1106 gpt-4-0613
```

This should give you an output on the terminal that contains something like the following:

```
Playing games: 100%|██████████████████████████████████| 20/20 [00:48<00:00,  2.41s/it]
```

If that is the case, output (transcripts of the games played) will have been written to 
`results/gpt-3.5-turbo-t0.0--gpt-3.5-turbo-t0.0/taboo` (in the main directory of the code).

Unfortunately, at the moment the code often fails silently, for example if model names are wrong, so make sure that you see the confirmation that the game actually has been played. Have a look at the file `clembench.log` if you suspect that something might be wrong.

You can get more information about what you can do with the `cli` script via:

```
python3 scripts/cli.py run --help
```

For example, you can use that script to get a more readable version of the game play jsons like so:

```
python3 scripts/cli.py transcribe -g taboo
```

After running this, the `results` directory will now hold html and LaTeX views of the transcripts.


To run other game masters individually use the following scripts. Note some games (privateshared) are single player and some games can be multiplayer (taboo, referencegame, imagegame, wordle)

```
python scripts/cli.py run -g privateshared -m gpt-3.5-turbo 
```

```
python scripts/cli.py run -g taboo -m gpt-3.5-turbo 
```

```
python scripts/cli.py run -g imagegame -m gpt-3.5-turbo 
```

```
python scripts/cli.py run -g referencegame -m gpt-3.5-turbo 
```

```
python scripts/cli.py run -g wordle -m gpt-3.5-turbo 
```


## Running the benchmark

Go into the project root and prepare path to run from cmdline

```
source prepare_path.sh
```

Then, run the wrapper script:

```
./pipeline_clembench.sh
```

Internally, this uses `run.sh` to run individual game/model combinations. Inspect the code to see how things are done.

## Running the evaluation

All details from running the benchmarked are logged in the respective game directories,
with the format described in ```logdoc.md```.

In order to generate the transcriptions of the dialogues, please run this command:

```
python3 scripts/cli.py transcribe
```

Or put a single game name (taboo, referencegame, imagegame, wordle, privateshared)

```
python3 scripts/cli.py transcribe -g taboo
```

Next, run this command to generate the scores of the dialogues:

```
python3 scripts/cli.py score
```

Or put a single game name (taboo, referencegame, imagegame, wordle, privateshared)

```
python3 scripts/cli.py score -g taboo
```

We provide an evaluation script at `evaluation/papereval.py` that produces a number of tables and visualizations for all games in the ```results/``` directory, which was used for the paper. To use this script, new models (their name abbreviation), metrics (their range) and game/model (their order) must be added manually to the constants in ```evaluation/evalutils.py```. Run the following to replicate the results in the paper or if you have new results:

```
python3 evaluation/papereval.py
```

If all you need is a table with the leaderboard results (% played, main score and clemscore for each game and model), you can run:

```
python3 evaluation/bencheval.py -p PATH_TO_RESULTS
```

where ```PATH_TO_RESULTS``` is your results folder. By default it will access ```./results/```.

The latest relies solely on the structure of the results directory, so it can be run with any games and models you have. The table will be saved into ```PATH_TO_RESULTS/results.csv```, with a copy in html.

Each game requires a custom analysis. The notebook ```howto_evaluate.ipynb``` shows examples of how to begin evaluating games. In particular, it generates tables with an overview of all metrics and also detailed by experiment. It also reproduces Figure 10 from the paper.

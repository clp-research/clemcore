### Updates
(March 2025): Version 2.0 of the benchmark has been released (see <https://clembench.github.io/> ). And some major refactorings are underway and will be released very soon. (Preview: The framework will become pip-installable, and the games that make the benchmark will get their own repo. (In fact, the code used for running 2.0 already lives in a separate repo: <https://github.com/clp-research/clemgames> .) Everything will become easier. Hopefully.)

(February 2024): We have updated the framework code. If you have written games using the initial release version, see [this guide](docs/howto_update_to_v1.md) on how to update your game.

# clembench: A Framework for the Systematic Evaluation of Chat-Optimized Language Models as Conversational Agents

The cLLM (chat-optimized Large Language Model, "clem") framework tests such models' ability to engage in games – rule-constituted activities played using language.
The framework is a systematic way of probing for the situated language understanding of language using agents.

This repository contains the code for setting up the framework and implements a number of games that are further discussed in 

> Chalamalasetti, K., Götze, J., Hakimov, S., Madureira, B., Sadler, P., & Schlangen, D. (2023). clembench: Using Game Play to Evaluate Chat-Optimized Language Models as Conversational Agents (arXiv:2305.13455). arXiv. https://doi.org/10.48550/arXiv.2305.13455

### Evaluation Results

On the [main project website](https://clembench.github.io) , under [leaderboard](https://clembench.github.io/leaderboard.html).

### Game details

see [games repository](https://github.com/clp-research/clemgames)
- A Simple Word Game: [taboo](docs/taboo.md)
- A Word-Guessing Game Based on Clues: [wordle](docs/wordle.md)
- Drawing Instruction Giving and Following: [image](docs/image.md)
- An ASCII Picture Reference Game: [reference](docs/reference.md)
- Scorekeeping: [private and shared](docs/privateshared.md)

# Using the clemcore CLI

The project is now pip installable.
This means that there is no need to checkout the repository, but you can simply install the packaged project as usual:
```
(myclem) pip install clemcore
```
(However, clemcore developers should checkout this repository and install from within the directory `pip install . -e`) 

Note that we highly recommend to perform the installation in a distinct virtual python environment, because there might be a lot of dependencies necessary depending on your use case.
Additional install options are:
```
(myclem) pip install clemcore[huggingface] # dependencies for the local hf backend
(myclem) pip install clemcore[vllm]        # dependencies for the local vllm backend
(myclem) pip install clemcore[slurk]       # dependencies for the slurk backend 
```

After the installation you will have access to the `clem` CLI tool. The main functions are:

```
(myclem) clem list games               # list the games available for a run
(myclem) clem list backends            # list the backends available for a run
(myclem) clem list models              # list the models available for a run
(myclem) clem run -g <game> -m <model> # runs the game benchmark; also transcribes and scores
(myclem) clem transcribe               # translates interactions into html files
(myclem) clem score                    # computes individual performance measures
(myclem) clem eval                     # computes overall performances measures; requires scores
```

Note that `clem` operates relative to the current working directory, that is, the directory it is called from.
This directory is what we call the workspace.
A workspace may look like this.

```
(optional) key.json
(optional) game_registry.json 
(optional) model_registry.json  
(optional) custom_api.py 
clemgames/
```

The files have the following functions:
- **key.json**: contains the secrets for the remote api calls; if this file does not exist, then `clem` looks into `~/.clemcore/`
- **game_registry.json**: allows to make additional game specifications useable for the runs. The game specifications must at least contain the `game_name`, `game_path` and `players` attribute. 
- **model_registry.json**: allows to add additional model specifications. This is specifically useful to run with models that have not been packaged yet. In addition, it allows to point model specification to custom backend names.
- **custom_api.py**: `clem` automatically discovers additional _api files placed into the cwd, so that users of the framework can run their own backends with the games.
- **clemgames/**: contains the game directories (with the game code) available for the benchmark runs

Note that, `clem` does now automatically discovers game directories that are at most 3-levels away from the `cwd`. 
To be discoverable, directories have to carry a `clemgame.json` (here a game path is not required, because `clem` automatically determines it).

## Use Case: Benchmarker

As a benchmarker you want to run multiple models for all games that constitute the benchmark.
Therefore, you will checkout the `clemgames` repository into a new workspace directory.
You will add the `key.json` to the workspace to access the backends.
In addition, you might need to add additional model entries that are not yet packaged to a `model_registry.json`.
Then you will run via the cli `clem run -g all -m model1` etc. or potentially use a batch script.
When not otherwise specified, then the results files will be stored in the cwd under `results`. 
Hence, a benchmarkers workspace directory might look as follows:

```
myworkspace
- clemgames/
- results/
- key.json 
- model_registry.json  
```

## Use Case: Game Developer

As a game developer you want to implement your own game to be run with `clem`.
You will use a typical clembench game project structure.
The game directory will become your workspace.
To make the game visible to `clem` you need to add a `clemgame.json` to the directory.
This file should specify at least the following
```
{
"game_name": "mygame",
"description": "A brief description of mygame",
"player": "single" | "two" | "multi",
"image": "none" | "single" | "multi",
"languages": ["en"]
}
```

To test your game with some packaged models, you will add a `key.json` and run the command `clem run -g mygame -m model` from within the game directory.
The results will be written into `results`.
To also get html transcripts you can run `clem transcribe -g mygame`.
Overall, a game developers workspace directory will possibly look as follows:

```
mygame
- in/
- resources/
- results/
- __init__.py
- master.py
- instancegenerator.py
- clemgame.json
- key.json   
```

## Use Case: Model Developer

As a model developer you want to test the performance of your custom model on the benchmark.
For this you will checkout the `clemgames` repository into your workspace directory.
In addition, you want to make your custom model available via the `model_registry.json`.
The entry should at least specify a name and a backend, e.g., `{"model_name":"mymodel", "backend":"mybackend"}`. 
The important thing to consider is that `clem` will try to locate all additional backend files in the workspace.
Therefore, one of this should match the backend specified in the registry, meaning that you will create an `mybackend_api.py` in the workspace.
This files mainly implements the `generate_response` method for the model and might specify how it is loaded.
Finally, you will run `clem -g all -m mymodel` from the workspace directory to run your model on all games.
The results will be written into the `results` directory.
Hence, a model developers workspace might look as follows:

```
myworkspace
- clemgames/
- results/
- model_registry.json
- mybackend_api.py  
```



We welcome you to contribute to or extend the benchmark with your own games and models. 
Please open a pull request in the respective repository. 
You can find more information on how to use the benchmark in the links below.

However, the following documentation needs still to be checked for up-to-dateness.

- [How to run the benchmark and evaluation locally](docs/howto_run_benchmark.md)
- [How to run the benchmark, update leaderboard workflow](docs/howto_benchmark_workflow.md)
- [How to add a new model](docs/howto_add_models.md)
- [How to add and run your own game](docs/howto_add_games.md)
- [How to integrate with Slurk](docs/howto_slurk.md)

This repository is tested on `Python 3.10+`
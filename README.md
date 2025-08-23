# clembench: A Framework for the Systematic Evaluation of Chat-Optimized Language Models as Conversational Agents

The cLLM (chat-optimized Large Language Model, "clem") framework allows researchers to easily evaluate the ability of large language models (LLM) by engaging them in games – rule-constituted activities played using language.

The framework is a systematic way of probing for the models' situated language understanding by framing them as agents, i.e., players which interfere with a game master in 1-to-1 conversations.

This repository contains `clemcore`, the core framework code used to implement and run the games discussed in our EMNLP paper:
> Chalamalasetti, K., Götze, J., Hakimov, S., Madureira, B., Sadler, P., & Schlangen, D. (2023).
> [clembench: Using Game Play to Evaluate Chat-Optimized Language Models as Conversational Agents.](https://aclanthology.org/2023.emnlp-main.689/)
> In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 11174–11219,
> Singapore. Association for Computational Linguistics.

**Clembench Repository**

While the paper is called clembench, because it presents the set of games we propose to evaluate the models capabilities, we decided to separate the framework code from the specific game implementations.
You can find the official set of benchmark games in the [clembench repository](https://github.com/clp-research/clembench).

**Clembench Results**

The results of running the benchmark on the games are uploaded to our [main project website](https://clembench.github.io).
The individual results for each run can be inspected via our [transcript browser](https://clembench.github.io/transcript-browser.html).
From the results we constitute a [leaderboard](https://clembench.github.io/leaderboard.html) which shows the performance of the most relevant models.

---

## Table of Contents

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [CLI Commands](#cli-commands)
- [Games](#games)
- [Models](#models)
- [Backends](#backends)
- [Contributing](#contributing)

---

## Overview

The **clemcore** framework provides a systematic way of assessing *situated language understanding* of large language models by framing them as agents in rule-governed games.

- **This repo (**clemcore**)** → core framework, installable via pip.
- [**clembench repo**](https://github.com/clp-research/clembench) → set of official benchmark games built on top of clemcore.
- [**Project website**](https://clembench.github.io) → results, transcript browser, and leaderboard.

| Component     | Purpose                                                |
| ------------- | ------------------------------------------------------ |
| **clemcore**  | Framework code, CLI (`clem`), backends, model registry |
| **clembench** | Collection of benchmark games for evaluation           |

---

## Quickstart

Install and run your first game in a fresh virtual environment:

```bash
# 1. Create and activate a virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 2. Install clemcore
pip install clemcore

# 3. Download clembench games and install requirements
export CLEMBENCH_HOME=/path/to/clembench
git clone https://github.com/clp-research/clembench $CLEMBENCH_HOME
pip install -r $CLEMBENCH_HOME/requirements.txt

# 4. List available games and try a dry-run with the mock model (programmatic player)
clem list games
clem run -g taboo -m mock mock

# 5. Run llama3-8b against the text-only benchmark (version 2 game instances)
clem run -g "{'benchmark': ['2.0']}" -m Meta-Llama-3.1-8B-Instruct

# 6. Perform a quantitative evaluation of the results
clem score && clem eval

# 7. Inspect episodes of game play for qualitative evaluation
clem transcribe
```

---

## Installation

We recommend to use Python 3.10 in a dedicated virtual environment:

```bash
sudo apt install python3.10 python3.10-venv python3.10-dev
python3.10 -m venv venv
source venv/bin/activate
```

Install clemcore:

```bash
pip install clemcore
```

Optional extras:

```
pip install clemcore[huggingface] # installs dependencies for using the local huggingface backend
pip install clemcore[vllm]        # installs dependencies for using the local vllm backend
pip install clemcore[slurk]       # installs dependencies for using the slurk backend 
```

Check installation:

```
clem --version 
# clem 3.3.2
```

---

## CLI Commands

After the installation you will have access to the `clem` CLI tool. 
The main functions are:

```
clem list games               # list the games available for a run
clem list backends            # list the backends available for a run
clem list models              # list the models available for a run
clem run -g <game> -m <model> # runs specified game using specified model
clem transcribe               # translates interactions into html files
clem score                    # computes individual performance measures
clem eval                     # computes overall performances measures; requires scores
```

---

## Games

The [clembench repository](https://github.com/clp-research/clembench) provides a set of ready‑to‑use games.
To make them available to the `-g <game-name>` option of `clem run`:
1. Clone `clembench` to a directory of your choice.
2. Either run `clem` from within that directory, **or** set the `CLEMBENCH_HOME` environment variable to point to it.

Alternatively, you can place a `game_registry.json` file in your current working directory that points to the benchmark folder:

```json
[{
  "benchmark_path": "path/to/clembench"
}]
```

To check the available games, run the following command:

```bash
clem list games
# Listing all available games (use -v option to see the whole specs)
# Found '26' game specs that match the game_selector='all'
# adventuregame:
#         Interactive Fiction clemgame
# cloudgame:
#         A simple game in which a player has to decide whether they see clouds
#         or not and a second player has to judge this response.
# ...
```

If you want to list only a subset of games (not all, but more than one), you can use the `-s` (selector) option:

```
clem list games -s "{'benchmark':['2.0']}"
# Listing all available games (use -v option to see the whole specs)
# Found '14' game specs that match the game_selector='{'benchmark': ['2.0']}'
# adventuregame:
#         Interactive Fiction clemgame
# codenames:
#         Codenames game between a cluegiver and a guesser
# ...
```

> **Note:** These selectors can also be passed to the `-g` option of the `clem run` command!
> 
To register custom games extent the `game_registry.json`. 
A minimal entry looks like this:

```json
{
  "game_name": "mygame",
  "game_path": "path/to/mygame",
  "description": "A brief description of mygame",
  "player": 1,
  "image": "none",
  "languages": ["en"]
}
```

---

## Models

A model implements an interface to generate a player's response given a message history during game play.

The `clemcore` package already comes with a huge variety of models registered in a bundled `model_registry.json`.
This makes them available for the `-m <model>` option of the `clem run` command.

To check the available list, run the following command:

```bash
clem list models | head
# Listing all available models by name (use -v option to see the whole specs)
# Found '215' registered model specs:
# slurk -> slurk (packaged)
# openchat -> openai_compatible (packaged)
# codellama-34b -> openai_compatible (packaged)
# Llama-3-70B-Instruct-Anyscale -> openai_compatible (packaged)
# Llama-3-70B-Together.ai -> openai_compatible (packaged)
# mistral-large-2411 -> openai_compatible (packaged)
# deepseek-v3 -> openai_compatible (packaged)
# deepseek-r1 -> openai_compatible (packaged)
```

When you want to add a model, then simply specify the model in a `model_registry.json` within the directory where you run `clem` CLI. 
A minimal model specification would define the model's name and the backend to be used:

```json
{
  "model_name": "custom_model",
  "backend": "custom_backend"
} 
```

The model will then show up in the listing:

```bash
clem list models | head
# Listing all available models by name (use -v option to see the whole specs)
# Found '216' registered model specs:
# custom_model -> custom_backend (/path/to/cwd/model_registry.json)
# slurk -> slurk (packaged)
# ...
```

> **Note:** Models defined by custom files in the current working directory always precede packaged models.

---

## Backends

The `clemcore` supports out-of-the box a variety of model providers.
These backends are responsible to provide the models (or agents) that are supposed to play the games.

To see the available backends run the following command:

```bash
clem list backends
# Listing all supported backends (use -v option to see full file path)
# Found '14' supported backends.
# Then you can use models that specify one of the following backends:
# anthropic (packaged)
# slurk (packaged)
# vllm (packaged)
# mistral (packaged)
# huggingface_local (packaged)
# openai_compatible (packaged)
# openai (packaged)
# google (packaged)
# huggingface_multimodal (packaged)
# llamacpp (packaged)
# cohere (packaged)
# alephalpha (packaged)
# _player_human (packaged)
# _player_programmed (packaged)
```

You can add your own backend simply by adding a `<backend-name>_api.py` with a `Backend` implementation into the directory where you call the `clem` CLI.
The backend will then show up in the listing:

```bash
clem list backends
# Listing all supported backends (use -v option to see full file path)
# Found '15' supported backends.
# Then you can use models that specify one of the following backends:
# custom_backend (cwd)
# anthropic (packaged)
# ...
```

> **Note:** Models defined by custom files in the current working directory always precede packaged models.

### Proprietary Backends

Proprietary models are often only provided through backends that connect to a remote API.
These remote APIs are usually protected by the use of API keys.

To make all remote API backends fully functional, you have to add your secure key to a `key.json`. Use the provided `key.json.template`:

- Rename the file to `key.json`
- Add your keys in the `api_key` entries
- Place the file in either the working directory or `~/.clemcore`

> **Note:** Keys defined in the current working directory always precede others

### OpenAI Compatible Backend

The openai compatible backend comes with an additional `base_url` field in the `key.json` which allows you to define a remote API that is compatible with the OpenAI client.
This comes in very handy, when you, for example, host your own models via a `vllm` server.

> **Note:** When using this backend you usually want to add your own model specifications (see above).

### Slurk Backend

The `clemcore` framework integrates with the [slurk experiment server](https://github.com/clp-research/slurk).
This "Chat Server for Dialogue Experiments and Data Collection" enables humans to play the games simply by using a browser.

Hence, for this to work, you have to set up a slurk server instance.
For testing purposes you can do this on your local machine using a docker container:

```
docker run -d -e FLASK_ENV=development -p 5000:80 slurk/server
```

Running in dev mode should start a slurk server on port `5000` and expose an api that is protected by the default `api_key`.
Now, similar to the openai compatible backend, `clem` must be informed about the location of the slurk host, by filling in the respective entries in the `key.json`:

```
    "slurk": {
        "api_key": "00000000-0000-0000-0000-000000000000", # default value
        "slurk_host": "http://localhost:5000"
      }
```

Finally, the slurk backend is fully functional and becomes available to `clem`.
Note that in terms of benchmarking `clemcore` frames the human player as a "model" backed up by the slurk backend.
Hence, the command to play a single-player game is:

```
clem run -g <game> -m slurk
```

This will set up everything and expose a clickable url in the console output which redirects to the game room on the slurk server:

```
2025-08-20 13:59:16,531 - clemcore.cli - INFO - http://localhost:5000/login?name=player&token=091aee66-eecb-4da4-88dc-a6680384be82
```

Notably, the first 8 characters of the login token act as the model name, e.g. `091aee66`, to distinguish between players in the results folder.

For multi-player games, the index of the model argument determines the role to be played.
Look up the specific game specification to find out which index maps to which role.
You can simply play against a supported model by using, for example:

```
clem run -g <game> -m slurk <model-name>
```

---

## Contributing

Framework developers that want to contribute to the clemcore framework should follow the following steps:

- Fork this repository via GitHub
- Clone the forked repository to your development machine
- Create a venv as mentioned above and install the project with `pip install -e .`
- Make sure that the venv folder is git-ignored
- Create a branch in the fork that is supposed to contain your changes
- Test you changes either by adding a test case or by installing the framework locally and running the CLI
- Commit and push your changes to the branch on your fork
- Create a pull request that aims to merge your branch with the main branch of this repository

---

This repository is tested on `Python 3.10`.
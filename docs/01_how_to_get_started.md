## Instructions on how to run existing games:

1. Create your working directory. Set up a python environment in your working directory and install the clemcore 
library. For support for specific backends, install clemcore with the `[huggingface]`, `[vllm]` or `[slurk]` flags, for 
example `pip install clemcore[huggingface]`.

2. Clone the [clemgames repository](https://github.com/clp-research/clemgames.git) into your working directory.

3. Create a `key.json` file in your working directory, matching the structure in `key.json.template`, easily done by 
copying `key.json.template` and removing the `.template` suffix. This is required even if you do not use remote API 
inference.

4. To evaluate a remote API model, add the required keys to `key.json` for either connecting to the model served by our 
group (see pinned post in mattermost channel CLEM-Club), or any remote API by creating an account (e.g., Groq, 
OpenAI, ...)

5. Downloading and running any (supported) model hosted on HuggingFace locally is covered by the clemcore framework. To 
set a custom HuggingFace hub cache directory to store the model files in, run 
`export HUGGINGFACE_HUB_CACHE=/data/>USERNAME>/huggingface_cache`) via terminal. You can check which models are 
supported with the CLI command `clem list models`. See the [model registry readme](model_backend_registry_readme.md) 
and the [adding models howto](howto_add_models.md) for details.

6. Install clemgame specific requirements listed in `clembench/requirements.txt` and the respective game directories. 
Running `pip install clembench/requirements.txt` is the simplest way to do this.

7. To run a game, use the `clem run` CLI command. For example, to run the Taboo clemgame with mock 
model responses, the CLI commmand is `clem run -g taboo -m mock`. Use `clem run -h` for more example calls.
* Run `clem list games` for all currently available/supported games.
* Run `clem list models` for all currently registered/supported models.
* Instances are expected to be stored in each game's directory under `in/instances.json`, if not specified otherwise 
using the `-i` argument for the `clem run` CLI command. For example, `clem run -g taboo -m mock -i instances_v2.0` will 
run the taboo game with mock model responses using instances stored in the `clembench/taboo/in/instances_v2.0.json`
file.
* Results will be stored in `./results`, if not specified otherwise using the `-r` argument for the `clem run` CLI 
command. For example, `clem run -g taboo -m mock -r testing` will run the taboo game with mock model responses and store
the result files in the `./testing` directory.
* Run information will be logged will be written to `./clembench.log`, and model-specific inference logs will be written 
to the `./logs` directory.

8. After the game/benchmark was run, use the `clem score` CLI command to score the results. For example, 
`clem score -g taboo` will score the results of the taboo game. If no `-g` argument is given, all games' results will be 
scored.
* The results directory to score defaults to `./results`, but can be set to a different path with the `-r` argument. For
example, `clem score -r testing` will score results stored in the `./testing` directory.

9. After all results have been scored, use `clem eval` to create the evaluation table, which calculates 
`clemscore, %Played and Success` for all games and models found in the results directory and creates an overview in 
`./results/results.[csv|html]`.
* The results directory to evaluate defaults to `./results`, but can be set to a different path with the `-r` argument. 
For example, `clem eval -r testing` will evaluate results stored in the `./testing` directory.

10. See the ´clembench/scripts´ folder for example scripts on how to run the whole pipeline.


## Instructions on how to add new games:
1. Follow the steps above, but instead of cloning the clemgames repository, create your own game repository (for 
compatibility, create it inside your working directory).
2. Create a `clemgame.json` file in your game's root directory `./YOURGAME` to allow clemcore to detect it. All keys in the following 
example must be present: 
```
{
  "game_name": "yourgame",
  "description": "Your new clemgame",
  "main_game": "yourgame",
  "players": 1,
  "image": "none",
  "languages": ["en"],
  "benchmark": [],
  "regression": "large",
  "roles": ["YourGamePlayer"]
}
```
Clemcore searches for `clemgame.json` files in subdirectories of your working directory and sibling directories of it up 
to a depth of three.
3. Create instances (see existing instancegenerator.py files in the game directories for examples) in `./YOURGAME/in`, 
storing required resources in `./YOURGAME/resources` and create the game master in `master.py` in `./YOURGAME`. See the 
[adding games readme](howto_add_games.md) and the [adding games howto](howto_add_games_example.ipynb) for more details.
* To develop the game structure, it can be helpful to first define custom responses in players that always answer 
according to the formal rules and can be run using `-m mock`.
* For evaluation, use `clemcore/clemeval.py` as a starting point and potentially extend it for game specific evaluation.
* To add your final game to the official collection, create a PR in the clemgames repo.
* For any required changes regarding the clemcore framework, open an issue/PR in the clemcore repo.

## Instructions on how to update games to the new framework version:
+ If you started developing your game before December 2024, and if it was not added to the official games yet, you need 
to update your game to the new framework version as described [here](howto_update_to_v2.md).
* Some games in `clemgames` still need to be updated (this is work in progress).

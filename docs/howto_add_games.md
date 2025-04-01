# The Lightweight Dialogue Game framework

### Preliminaries

If you're completely new to this, it might make sense to look at two Jupyter notebooks that we provide here, which explain how to set up new games a bit more from scratch:

- [How to Prototype Games](https://github.com/clp-research/clembench/blob/main/docs/howto_prototype_games.ipynb) explains how to use our backends to make first tests of prompts with a variety of LLMs easy to do, and then how to prototype your game loop.
- [How to Add Games](https://github.com/clp-research/clembench/blob/main/docs/howto_add_games_example.ipynb) takes this further and shows how you get from the prototype to an implementation that can use all the clembench infrastructure for running the game repeatedly with different instances and models.

### Introduction

The benchmark is run for a particular game -- for example the taboo game -- using the follow command:  

```
python3 scripts/cli.py run -g taboo -m gpt-3.5-turbo-1106
```

_Note: when only a single model for a 2-player game is given, then clem will use this model for both players!_ 

As taboo is a game of two players (a clue giver and a guesser) we could theoretically also let two different
models play the game which would look like:

```
python3 scripts/cli.py run -g taboo -m gpt-3.5-turbo-1106 gpt-4-0613
```

### GameBenchmark class

When the command is executed then the `run` routine in `benchmark.py` 
will determine the game code that needs to be invoked.
For this the benchmark code loads all **subclasses** of type `GameBenchmark` and calls `setup()` 
on them. The setup method already loads the game instances (`self.load_json("in/instances.json")`). 
After this each game benchmark **subclass** is asked if it applies to the given game name, here `taboo`.  

Therefore, such a **subclass** has to be provided with a specific game name 
for each game to be run in the benchmark, for example for taboo:

```
class TabooGameBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "Taboo game between two agents where one has to describe a word for the other to guess."

    def create_game_master(self, experiment: Dict, player_backends: List[str]) -> GameMaster:
        return Taboo(experiment, player_backends)
        
    def is_single_player(self) -> bool:
        return False
```

The respective subclass simply provides the `GAME_NAME=taboo` and the `GameBenchmark` super class is taking care of most
of the necessary plumbing and executes the main logic for a benchmark run (calling the game master, loading files etc.).

Aside: The return value of `get_description` is shown for the `python3 scripts/cli.py ls` command.

Then the benchmark code checks if your game is single or multiplayer game (the default is multi-player), 
so that the `-m gpt-3.5-turbo-1106` option is properly handled. 
Then the `run(dialog_pair,temperature)` method is called which is already implemented by `GameBenchmark`.
This is when the `GameMaster` becomes relevant (which should be returned by your `create_game_master()` factory method).

### GameMaster class

Now for each experiment in the `instances.json` -- that has been loaded on_setup() -- the game benchmark code 
applies the given dialog pair (or if not given tries to determine the dialogue pair from the instance information).

Aside: There is also the option to provide multiple dialogue pairings in the experiments in `instances.json`. 
Therefore, the code must check again, if these pairing align to the nature of the game (single or multiplayer).

Each experiment represents a specific condition for the game, for example the assumed difficulty of the game instances
and holds the actual game instances themselves. Then for each game instance a `GameMaster` is created 
by using the `self.create_game_master()` method of the `GameBenchmark`. The `GameMaster` is in charge of actually 
playing a single instance of the game. 
For taboo this would be a target word to be guessed and the words that are not allowed to be said.
The relevant code looks as follows:

```
try:
   game_master = self.create_game_master(experiment_config, dialogue_pair)
   game_master.setup(**game_instance)
   game_master.play()
   game_master.store_records(game_id, game_record_dir)
except Exception:  # continue with other episodes if something goes wrong
   self.logger.exception(f"{self.name}: Exception for episode {game_id} (but continue)")
```

We see that game master receives the game instance information on `setup()`. 
Then coordinates the `play()` of the actual game. And finally calls `store_records` to stores 
the interactions between the players and the game master during the turns in the `game_record_dir` 
(this directory is prepared by the `GameBenchmark`).

### Overview

These are the important classes and methods to be implemented for your own game.

A`MyGameBenchmark` that extends `GameBenchmark` and implements:
- `def __init__(self)` with call to `super().__init__(GAME_NAME)`
- `def get_description(self)` that returns a description
- `def is_single_player(self) -> bool` that determines if one player is sufficient
- `def create_game_master(self, experiment: Dict, player_backends: List[str]) -> GameMaster` that returns `MyGameMaster` for my game

A`MyGameMaster` that extends `GameMaster` and implements:
- `def __init__(self, name: str, experiment: Dict, player_backends: List[str] = None):` that receives the experiment information and the players that play the game. These can be simply delegated to `super()`.
- `def setup(self, **game_instance)` which sets the information you specify in `instances.json`
- `def play(self)` that executes the game logic and performs the turns in the game
- `def compute_scores(self, episode_interactions: Dict)` that is called later when the user executes the `python3 scripts/cli.py score taboo` command

Note that the `store_records` method is already implemented by `GameRecorder` 
and every `GameMaster` extends that class. This means that the method must not be implemented.

### DialogueGameMaster

Now we can see that `MyGameMaster` has all the freedom to implement `play()` which might be in some cases a nice thing.
In other cases we already know that the gameplay will be executed in turns of for example two players.
For these cases you can extend from `DialogueGameMaster` a more conrete subclass of `GameMaster`.

The dialogue game master defines a play routine that is as follows:

```python
 def play(self) -> None:
    self._on_before_game()
    while self._does_game_proceed():
        self.log_next_round()  # not sure if we want to do this always here (or add to _on_before_turn)
        self._on_before_round(self.current_round)
        self.logger.info(f"{self.name}: %s turn: %d", self.name, self.current_round)
        for player in self.__player_sequence():
            if not self._does_game_proceed():
                break  # potentially stop in between player turns
            # GM -> Player
            history = self.context_for_player[player.descriptor]
            assert history, f"messages history must not be empty for {player.descriptor}"

            last_entry = history[-1]
            assert last_entry["role"] != "assistant", "Last entry should not be assistant "
            "b.c. this would be the role of the current player"
        message = last_entry["content"]

        action = {'type': 'send message', 'content': message}
        self.log_event(from_='GM', to=player.descriptor, action=action)

        _prompt, _response, response_message = player(history, self.current_round)

        # Player -> GM
        action = {'type': 'get message', 'content': response_message}
        self.log_event(from_=player.descriptor, to="GM", action=action, call=(_prompt, _response))

        # GM -> GM
        self.__validate_parse_and_add_player_response(player, response_message)
    self._on_after_round(self.current_round)
    self.current_round += 1


self._on_after_game()
```

Let's have a look on this routine. As long as the game proceeds (`_does_game_proceed()`):

**GM -> Player.**
At a player's turn, the player receives its view on the history of messages (`messages_by_names`) and the last
messages is logged (`log_event`) as a `GM->Player` event in the interactions log. 
Then player is asked to create a response based on the history and the current turn index.

**Player -> GM.**
The player response is received and logged as a `Player->GM` event in the interactions log.

**GM -> GM.**
The game master received the player response and validates its content. When the 
validation is successful then the response is added to all player's history and 
the next player's turn is performed with the same procedure.

This shows that the logging is already done systematically when using the `DialogueGameMaster`.
Still, there are several hooks for you to customize the gameplay:

- `def _on_setup(self, **kwargs)` which must be implemented. Use `add_player()` here to add the players.
- `def _does_game_proceed(self) -> bool` which must be implemented. Decides if the game can continue.
- `def _validate_player_response(self, player: Player, utterance: str) -> bool` to decide if an utterance should be added. This is also the place to check for game end conditions. 
- `def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]` to decide if a response utterance should be modified. If not simply return the utterance.
        When a modified utterance and a true value is returned, then a 'parse' event is logged.
- `def _after_add_player_response(self, player: Player, utterance: str)` to add the utterance to other player's history, if necessary.
        To do this use the method `add_user_message(other_player,utterance)`.
- the general game hooks `_on_before_game()` and `_on_before_game()`
- the general turn hooks `_on_before_turn(turn_idx)` and `_on_after_turn(turn_idx)`

Overall the game master acts here as a moderator between the players and the players actually never directly talk to each other.

For the `taboo` game we use the setup hook to set instance specific values and
to setup the `WordDescriber` and `WordGuesser` which are the `Player` for the game.
The player could also be LLMs defined by the `player_backends` descriptor string.

```python
 def _on_setup(self, **game_instance):
    logger.info("_on_setup")
    self.game_instance = game_instance

    self.describer = WordDescriber(self.player_models[0], self.max_turns)
    self.guesser = WordGuesser(self.player_models[1])

    self.add_player(self.describer)
    self.add_player(self.guesser)
```

We use the general game hook to set the initial prompts for both players

```python
def _on_before_game(self):
  self.add_user_message(self.describer, self.describer_initial_prompt)
  self.add_user_message(self.guesser, self.guesser_initial_prompt)
```

Then we must decide if the guessing should continue like

```python
 def _does_game_proceed(self):
    if self.invalid_response:
        self.log_to_self("invalid format", "abort game")
        return False
    if self.clue_error is not None:
        return False
    if self.current_round >= self.max_turns:
        self.log_to_self("max turns reached", str(self.max_turns))
        return False
    return True
```

And we have to check if the player response is actually in the valid format:

```python
def _validate_player_response(self, player, utterance: str) -> bool:
  if player == self.guesser:
      if not utterance.startswith("GUESS:"):
          self.invalid_response = True
          return False
  if player == self.describer:
      if not utterance.startswith("CLUE:"):
          self.invalid_response = True
          return False
      errors = check_clue(utterance, self.target_word, self.related_words)
      if errors:
          error = errors[0]
          self.clue_error = error
          return False
  self.log_to_self("valid format", "continue")
  return True
```

We see that this is also the place to potentially detect violations of the game rules.
Now we can also modify the message and for example log the responses without the prefixes.

```python
def _on_parse_response(self, player, utterance: str) -> Tuple[str, bool]:
  if player == self.guesser:
      utterance = utterance.replace("GUESS:", "")
      self.guess_word = utterance.lower()
      self.log_to_self("guess", self.guess_word)
  if player == self.describer:
      utterance = utterance.replace("CLUE:", "")
      self.log_to_self("clue", utterance)
  return utterance, False
```

The (possibly modified) response is then automatically added the player's history which is acting.
Still, for two-player games we have to add the response to the history of the other player as well.

```python
def _after_add_player_response(self, player, utterance: str):
    if player == self.describer:
        utterance = f"CLUE: {utterance}."
        self.add_user_message(self.guesser, utterance)
    if player == self.guesser:
        if self.guess_word != self.target_word:
            utterance = f"GUESS: {self.guess_word}."
            self.add_user_message(self.describer, utterance)
```

Finally, we need to use the general turn method to additionally log the initial prompt for the second player 
and not only the most recent one (as automatically done by the `DialogueGameMaster`).

```python
def _on_before_turn(self, turn_idx: int):
    if turn_idx == 0:
        self.log_message_to(self.guesser, self.guesser_initial_prompt)
```

### GameResourceLocator class

Note that the game masters are subclasses of the game resource locator.
This class provides methods to access, load and store files from within the game directory.

You should access resource only via the game resource locator! The locator knows how to refer to them.
For example use: `gm.load_json("my_file")` which is located directly at your game directory `game/my_file.json`.
You can access subdirectories by giving `gm.load_json("sub/my_file")` in `game/sub/my_file.json`.

The expected game folder structure would be as follows:
```
games
  ├──mygame
  │     ├── in
  │     │   └── instances.json
  │     ├── resources
  │     │   └── initial_prompt.template
  │     ├── instancegenerator.py
  │     └── master.py
  ...
```

The resource locator tries to load files from the respective `mygame` directory in the games folder.

### Player class

A `Player` object receives `messages` and returns a textual response.
A player generates this response either as a `_api_response()`
(calling a deployed cLLM) or by implemented behavior in `_custom_response()`.

For example, the taboo game guesser agent can be implemented as a player that can be a cLLM with a static response that always guesses the word "pear":

```python
from clemgame.clemgame import Player

class WordGuesser(Player):

   def __init__(self, model_name):
      super().__init__(model_name)

   def _custom_response(self, messages, turn_idx):
      # mock response
      return f'Pear'
```

### GameInstanceGenerator class

In order to let agents play a game, you need a description that instantiate single episodes.
For example, in the taboo game, each episode is played with a specific target word that also comes with a list of other, related and forbidden words.

The clemgame framework provides a `GameInstanceGenerator` class that you can use to generate full instances that also include initial prompts for the models and other meta information for running experiments.

For example, in the taboo game, we
- use word lists of 3 different frequency levels low/medium/high
- want to test 3 LLMs (taboo is played between 2 cLLMs)
- we fix the maximum number of turns to `N_GUESSES`
- we generate a fixed number of instances, `N_INSTANCES`
```python
from clemgame.clemgame import GameInstanceGenerator

N_INSTANCES = 20  # how many different target words; zero means "all"
N_GUESSES = 3  # how many tries the guesser will have
N_REATED_WORDS = 3
LANGUAGE = "en"

class TabooGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__("taboo")

    def on_generate(self):
        player_assignments = list(itertools.permutations([OpenAI.MODEL_GPT_35, Anthropic.MODEL_CLAUDE_13]))
        for difficulty in ["low", "medium", "high"]:

            # first choose target words based on the difficultly
            fp = f"resources/target_words/{LANGUAGE}/{difficulty}_freq_100"
            target_words = self.load_file(file_name=fp, file_ending=".txt").split('\n')
            if N_INSTANCES > 0:
                assert len(target_words) >= N_INSTANCES, \
                    f'Fewer words available ({len(target_words)}) than requested ({N_INSTANCES}).'
                target_words = random.sample(target_words, k=N_INSTANCES)

            # use the same target_words for the different player assignments
            experiment = self.add_experiment(f"{difficulty}_{LANGUAGE}", dialogue_partners=player_assignments)
            experiment["max_turns"] = N_GUESSES

            describer_prompt = self.load_template("resources/initial_prompts/initial_describer")
            guesser_prompt = self.load_template("resources/initial_prompts/initial_guesser")
            experiment["describer_initial_prompt"] = describer_prompt
            experiment["guesser_initial_prompt"] = guesser_prompt

            for game_id in tqdm(range(len(target_words))):
                target = target_words[game_id]

                game_instance = self.add_game_instance(experiment, game_id)
                game_instance["target_word"] = target
                game_instance["related_word"] = []

                if len(game_instance["related_word"]) < N_REATED_WORDS:
                    print(f"Found less than {N_REATED_WORDS} related words for: {target}")
```

This will then generate game instances as a json file at `games/taboo/in/instances.json`

### Adding your own game

To add your own game, create a submodule in `games` with the name of your game, for example `games.hellogame`.

Add to the module a `master.py` that implements the `GameMaster`.

### Running experiments with your game

```
python3 scripts/cli.py run -g hellogame -m gpt-3.5-turbo-1106 [-e greet_en]
```

Note: With -e you can specify specific experiments to run.

This will create a results folder in the project root as follows:

```
results
└── gpt-3.5-turbo-1106-t0.0--gpt-3.5-turbo-1106-t0.0
    └── hellogame
        └── 0_greet_en
            ├── episode_0
            │ ├── instance.json
            │ ├── interaction.json
            │ └── transcript.html
            ├── episode_1
            │ ├── instance.json
            │ ├── interaction.json
            │ └── transcript.html
            │ ...
            └── experiment_greet_en.json
```

The top level is `results` followed by directories that mention the involved model (pairings).

The model (pairing) sub-folders will contain a directory structure for each experiment
and the experiments episodes (game plays).

The episodes are defined by the game instances (from the `instances.json`) and
contain the instance parameters `instance.json`, an `interaction.json` and a nice human-viewable `transcript.html`.

The experiment folder also contains a `experiment_name.json` that contains the run parameters.

# Troubleshooting

### AssertionError: messages history must not be empty for Player

When using the `DialogueGameMaster`, then here the framework prevents a call to the remote API with an empty message
history.

1. Maybe you forgot to add the initial prompt to the players messages in `_on_before_game()`.
   For this use `self.add_user_message(<player>, prompt)`

2. You forgot to add the response of the preceding player to the
   message history of the current player in `_after_add_player_response(other_player, utt)`.
   For this use `self.add_user_message(current_player, utt)`

## Huggingface Prototyping Check Methods
The huggingface-local backend offers two functions to check messages lists that clemgames might pass to the backend 
without the need to load the full model weights. This allows to prototype clemgames locally with minimal hardware demand
and prevent common issues. See the [model registry readme](model_backend_registry_readme.md) for `ModelSpec`.
### Messages Checking
The `check_messages` function in `backends/huggingface_local_api.py` takes a `messages` list and a `ModelSpec` as 
arguments.  
It will print all anticipated issues with the passed messages list to console if they occur. It also applies the given 
model's chat template to the messages as a direct check. It returns `False` if the chat template does not accept the 
messages and prints the outcome to console.
### Context Limit Checking
The `check_context_limit` function in `backends/huggingface_local_api.py` takes a `messages` list and a `ModelSpec` 
as required arguments. Further arguments are the number of tokens to generate `max_new_tokens: int` (default: `100`), 
`clean_messages: bool` (default: `False`) to apply message cleaning as the generation method will, and `verbose: bool` 
(default: `True`) for console printing of the values.  
It will print the token count for the passed messages after chat template application, the remaining number of tokens
(negative if context limit is exceeded) and the maximum number of tokens the model allows as generation input.  
The method returns a tuple with four elements:  
- `bool`: `True` if context limit was not exceeded, `False` if it was.
- `int`: number of tokens for the passed messages.
- `int`: number of tokens left in context limit.
- `int`: context token limit.  
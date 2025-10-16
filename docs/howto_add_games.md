# The Lightweight Dialogue Game framework

### Preliminaries

If you're completely new to this, it might make sense to look at two Jupyter notebooks that we provide here, which 
explain how to set up new games a bit more from scratch:

- [How to Prototype Games](https://github.com/clp-research/clembench/blob/main/docs/howto_prototype_games.ipynb) explains how to use our backends to make first tests of prompts with a variety of LLMs 
easy to do, and then how to prototype your game loop.
- [How to Add Games](https://github.com/clp-research/clembench/blob/main/docs/howto_add_games_example.ipynb) takes this further and shows how you get from the prototype to an implementation that can use 
all the clemcore infrastructure for running the game repeatedly with different instances and models.

### Introduction

The benchmark is run for a particular game -- for example the taboo game -- using the follow command:  

```
clem run -g taboo -m gpt-3.5-turbo-1106
```

_Note: when only a single model for a 2-player game is given, then clemcore will use this model for both players!_ 

As taboo is a game of two players (a clue giver and a guesser) we could theoretically also let two different
models play the game which would look like:

```
clem run -g taboo -m gpt-3.5-turbo-1106 gpt-4-0613
```

### GameBenchmark class

When the command is executed then the `run` routine in `benchmark.py` will determine the game code that needs to be 
invoked. For this the benchmark code loads all **subclasses** of type `GameBenchmark` and calls `setup()` on them. The 
setup method already loads the game instances (`self.load_json("in/instances.json")`). After this each game benchmark 
**subclass** is asked if it applies to the given game name, here `taboo`.  

Therefore, such a **subclass** has to be provided with a specific game name for each game to be run in the benchmark, 
for example for taboo:

```
class TabooGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return Taboo(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return TabooScorer(self.game_name, experiment, game_instance)
```

The respective subclass simply provides the game's `GameSpec` and the `GameBenchmark` super class is taking care of most
of the necessary plumbing and executes the main logic for a benchmark run (calling the game master, loading files etc.).

Then the benchmark code checks if your game is single or multiplayer game (the default is multi-player), so that the 
`-m gpt-3.5-turbo-1106` option is properly handled.  
Then the `run(dialog_pair,temperature)` method is called which is already implemented by `GameBenchmark`.  
This is when the `GameMaster` becomes relevant (which should be returned by your `create_game_master()` factory method).

### GameMaster class
Now for each experiment in the `instances.json` -- that has been loaded `on_setup()` -- the game benchmark code 
applies the given dialog pair (or if not given tries to determine the dialogue pair from the instance information).

Aside: There is also the option to provide multiple dialogue pairings in the experiments in `instances.json`. 
Therefore, the code must check again, if these pairing align to the nature of the game (single or multiplayer).

Each experiment represents a specific condition for the game, for example the assumed difficulty of the game instances
and holds the actual game instances themselves. Then for each game instance a `GameMaster` is created 
by using the `self.create_game_master()` method of the `GameBenchmark`. The `GameMaster` is essentially in charge of 
actually playing a single instance of the game. (This is eventually done through method calls by the runner script.)  
For taboo this would be a target word to be guessed and the words that are not allowed to be said.  
The following is an abbreviation of the relevant code:

```
try:
   game_master = self.create_game_master(experiment_config, dialogue_pair)
   game_master.setup(**game_instance)
   done = False
   while not done:
       player, context = game_master.observe()
       response = player(context)
       done, info = game_master.step(response)
except Exception:  # continue with other instances if something goes wrong
   message = f"{game_benchmark.game_name}: Exception for instance {game_instance['game_id']} (but continue)"
   module_logger.exception(message)
```

We see that game master receives the game instance information on `setup()`, and then continuously updates the player's 
context with `observe()`, generates a response from the player (`player(context)`) and processes it to change the game 
state (`game_master.step(response)`) to play the game. Record keeping calls are omitted here, as the underlying 
`GameRecorder` takes care of them. See `clemcore/clemgame/runners/sequential.py` for the full code referenced here.

### Overview

These are the important classes and methods to be implemented for your own game.

A `MyGameBenchmark` that extends `GameBenchmark` and implements:
- `def __init__(self, game_spec: GameSpec)` with call to `super().__init__(game_spec)`
- `def create_game_master(self, experiment: Dict, player_models: List[str]) -> GameMaster` that returns `MyGameMaster` 
for my game
- `def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer` that returns `MyGameScorer` for 
my game

A `MyGameMaster` that extends `GameMaster` and implements:
- `def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[str] = None):` that receives the 
experiment information and the players that play the game. These can be simply delegated to `super()`.
- `def setup(self, **game_instance)` which sets the information you specify in `instances.json`
- `def observe(self)` which updates player context
- `def step(self)` that executes the game logic and performs the turns in the game
- NOTE: These are `GameMaster` base class methods and should not be used directly, instead a set of hook abstract 
methods should be implemented that are then called by these methods. See the [DialogueGameMaster section](#dialoguegamemaster).

A `MyGameScorer` that extends `GameScorer` and implements:
- `def compute_round_score(self, round_idx, round_events: List[Dict])` that calculates scores for a round of the game
- `def compute_episode_scores(self, interactions: Dict)` that calculates overall episode scores and must include the 
game's main BENCH_SCORE
- the scorer is called later when the user executes the `clem score taboo` command

Note that the `store_records` method is already implemented by `GameRecorder` and every `GameMaster` extends that class. 
This means that the method must not be implemented. In general, you only need to take care of logging your game's 
specific events and scores, while standard clemcore score recording is already taken care of by the framework code. 

### DialogueGameMaster

Now we can see that `MyGameMaster` has all the freedom to implement methods involved in playing the game which might be 
in some cases a nice thing.  
In other cases we already know that the gameplay will be executed in turns of, for example, two players.  
For these cases you can extend from `DialogueGameMaster` a more concrete subclass of `GameMaster`.

The DialogueGameMaster base class includes fully implemented `setup()`, `observe()` and `step()` methods:
```python
@final
def setup(self, **kwargs):
    """Load resources and prepare everything to play the game.
    Needs to log the players dictionary via self.log_players(players_dict).
    Intended to be left as-is by inheriting classes. Implement game-specific setup functionality in the _on_setup
    method.
    Called by the game's GameBenchmark run method for each game instance.
    Args:
        kwargs: Keyword arguments used to set up the GameMaster instance. This is usually a game instance object
            read from the game's instances.json.
    """
    self._on_setup(**kwargs)
    self._current_player = self.get_players()[self._current_player_idx]
    self._on_before_game()
    self.started = True
    self._on_before_round()

@final
def observe(self) -> Tuple[Player, Dict]:
    """
    Observe the current player context.
    Returns:
        Current Player object, current player context
    """
    player = self.current_player
    context = self.get_context_for(player)
    return player, context

@final
def step(self, response: str) -> Tuple[bool, Dict]:
    """
    Verifies the response and transitions the game by applying the current player's response for the turn.

    Args:
        response: The response (verbal action) of the current player.
    Returns:
        Bool determining if game is done, info about the processed game step
    """
    try:
        parsed_response = self._parse_response(self.current_player, response)  # throws ParseError
        self._advance_game(self.current_player, parsed_response)  # throws GameError
    except ParseError as error:
        self.count_request_violation()
        self._on_parse_error(error)
    except GameError as error:
        self._on_game_error(error)

    self.info["turn_score"] = self.compute_turn_score()
    self.info["turn_feedback"] = self.get_turn_feedback()

    # determine if the current player should pass the turn to the next player or get another turn:
    if self._should_pass_turn():  # True = move on to next player
        self._current_player = self._next_player()

    if self._start_next_round():
        self._on_after_round()
        self.current_round += 1  # already increment here b.c. _does_game_proceed might rely on it

    done = not self._does_game_proceed()
    if done:
        self._on_after_game()
        self.log_game_end()
        self.info["episode_score"] = self.compute_episode_score()
        for player in self.get_players():
            player.reset()
    elif self._start_next_round():  # prepare next round only when game has not ended yet
        self.__prepare_next_round()

    info = deepcopy(self.info)
    self.info = {}  # reset info after each step
    return done, info
```
These methods should not be changed for your game implementation. The methods called by `setup()` and `step()` are 
instead used to implement game-specific functionality.  
The methods that **must** be implemented for a working DialogueGameMaster subclass are:  
- `_on_setup(self, **kwargs)` has to contain the game-specific setup, based on the structure and content of your instances
- `_parse_response(self.current_player, response)` processes the player's response, checking it for game rule conformity 
and extracting game-relevant response content to return
- `_advance_game(self.current_player, parsed_response)` uses the processed player response to update the game state
- `compute_turn_score()` calculates a score for the current turn, while mandatory, it is only used for reinforcement 
learning (PlayPen)
- `compute_episode_score()` calculates a score for the entire, while mandatory, it is only used for reinforcement 
learning (PlayPen), and is not called as part of the normal game play loop
- `_does_game_proceed()` determines if the game play loop should continue, enforcing game rules

DialogueGameMaster assumes that for each *round*, each player takes at least one *turn*, meaning that *round* and *turn* 
are different. Only if there is a single player, giving only a single response that is used to advance the game, round 
and turn are conceptually the same - however, records and thus eventual scoring are round-based, so the difference 
between round and turn is important to anticipate. Overall the game master acts as a moderator between the players and 
the players actually never directly talk to each other.  
For the taboo example, in each round the word describer takes a turn describing the target word and the guesser takes a 
turn guessing, then the next round starts.

There are many methods involved in the processing of the game step, which are already implemented with minimal 
placeholder functionality in the DialogueGameMaster base class and are intended to be modified for a specific game.  
Below are the taboo-specific method implementations.

For the `taboo` game we use the setup hook to set instance specific values and to set up the `WordDescriber` and 
`WordGuesser` which are the `Player` subclasses for the game. The players use the `Model`s (LLMs, humans or 
programmatic) defined by the `player_models` argument. Adding the players in this order is crucial, as they are iterated 
over in the order they were added.

```python
def _on_setup(self, **game_instance):
    self.game_instance = game_instance

    self.target_word = game_instance["target_word"]
    self.related_words = game_instance["related_word"]

    describer_initial_prompt = self.experiment["describer_initial_prompt"]
    describer_initial_prompt = describer_initial_prompt.replace("$TARGET_WORD$", self.target_word)
    rel_words = f"- {self.related_words[0]}\n- {self.related_words[1]}\n- {self.related_words[2]}"
    describer_initial_prompt = describer_initial_prompt.replace("$REL_WORD$", rel_words)
    describer_initial_prompt = describer_initial_prompt.replace("$N$", str(self.max_rounds))

    guesser_initial_prompt = self.experiment["guesser_initial_prompt"]
    guesser_initial_prompt = guesser_initial_prompt.replace("$N$", str(self.max_rounds))

    self.describer = WordDescriber(self.player_models[0])
    self.guesser = WordGuesser(self.player_models[1])

    self.add_player(self.describer, initial_context=describer_initial_prompt)
    self.add_player(self.guesser, initial_prompt=guesser_initial_prompt)

    self.invalid_response = False
    self.clue_error = None
    self.guess_word = None
```

Then we must decide if the guessing should continue like

```python
def _does_game_proceed(self):
    """Proceed as long as the word hasn't been guessed and the maximum length isn't reached.
    """
    if self.is_terminal():
        if self.is_aborted():
            self.log_to_self("invalid format", "abort game")
        if self.is_clue_error():  # stop game if clue is wrong (for now)
            self.log_to_self("invalid clue", self.clue_error["message"])
        if self.is_turn_limit_reached():
            self.log_to_self("max rounds reached", str(self.max_rounds))
        if self.is_success():
            self.log_to_self("correct guess", "end game")
        return False
    return True
```

NOTE: on_valid_player_response and validate_player_response are now to be part of _parse_response

And we have to check if the player response is actually in the valid format:

```python
def _validate_player_response(self, player: Player, utterance: str) -> bool:
    if player == self.guesser:
        # validate response format
        if not utterance.startswith("GUESS:"):
            self.invalid_response = True
            return False
        self.log_to_self("valid response", "continue")
        # extract guess word
        guess_word = utterance.replace("GUESS:", "")
        guess_word = guess_word.strip()
        guess_word = guess_word.lower()
        guess_word = string_utils.remove_punctuation(guess_word)
        self.guess_word = guess_word.lower()
        self.log_to_self("valid guess", self.guess_word)
    if player == self.describer:
        # validate response format
        if not utterance.startswith("CLUE:"):
            self.invalid_response = True
            return False
        self.log_to_self("valid response", "continue")
        # validate clue
        clue, errors = check_clue(utterance, self.target_word, self.related_words, return_clue=True)
        if errors:
            error = errors[0]  # highlight single error
            self.clue_error = error
            return False
        self.log_to_self("valid clue", clue)
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

Finally, we need to use the general turn method to additionally log the initial prompt for the second player and not 
only the most recent one (as automatically done by the `DialogueGameMaster`).

```python
def _on_before_turn(self, turn_idx: int):
    if turn_idx == 0:
        self.log_message_to(self.guesser, self.guesser_initial_prompt)
```




The DialogueGameMaster base class defines a play routine that is as follows:

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
        self.__validate_parse_and_add_player_response(player, response_message)  # this odd method really existed sometime...?! not mentioned below either...
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
- `def _validate_player_response(self, player: Player, utterance: str) -> bool` to decide if an utterance should be 
added. This is also the place to check for game end conditions. 
- `def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]` to decide if a response utterance 
should be modified. If not simply return the utterance.
        When a modified utterance and a true value is returned, then a 'parse' event is logged.
- `def _after_add_player_response(self, player: Player, utterance: str)` to add the utterance to other player's history, 
if necessary.
        To do this use the method `add_user_message(other_player,utterance)`.
- the general game hooks `_on_before_game()` and `_on_before_game()`
- the general turn hooks `_on_before_turn(turn_idx)` and `_on_after_turn(turn_idx)`
- `def compute_response_score(self, response, context)` to calculate a score for individual responses (used for potential reinforcement learning, not necessarily benchmark scoring)
- `def compute_episode_score(self)` that calculates a score for the entire episode (used for potential reinforcement learning, not necessarily benchmark scoring)

Overall the game master acts here as a moderator between the players and the players actually never directly talk to 
each other.

For the `taboo` game we use the setup hook to set instance specific values and to set up the `WordDescriber` and 
`WordGuesser` which are the `Player`s for the game. The players could also be LLMs defined by the `player_models` 
descriptor string.

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

Finally, we need to use the general turn method to additionally log the initial prompt for the second player and not 
only the most recent one (as automatically done by the `DialogueGameMaster`).

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
mygame
   ├── in
   │   └── instances.json
   ├── resources
   │   └── initial_prompt.template
   ├── instancegenerator.py
   ├── clemgame.json
   └── master.py
  ...
```

The resource locator tries to load files from the respective `mygame` directory.

### Player class

A `Player` object receives `messages` and returns a textual response.  
A player generates this response either as a `_api_response()` (calling a deployed cLLM) or by implemented behavior in 
`_custom_response()`.

For example, the taboo game guesser agent can be implemented as a player that can be a cLLM with a static response that 
always guesses the word "pear":

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
For example, in the taboo game, each episode is played with a specific target word that also comes with a list of other, 
related and forbidden words.

The clemgame framework provides a `GameInstanceGenerator` class that you can use to generate full instances that also 
include initial prompts for the models and other meta information for running experiments.

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

To add your own game, create a module with the name of your game, for example `hellogame`.

Add to the module a `master.py` that implements the `GameMaster` and a `clemgame.json`.

### Running experiments with your game

```
clem run -g hellogame -m gpt-3.5-turbo-1106 [-e greet_en]
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

The model (pairing) sub-folders will contain a directory structure for each experiment and the experiments episodes 
(game plays).

The episodes are defined by the game instances (from the `instances.json`) and contain the instance parameters 
`instance.json`, an `interaction.json` and a nice human-viewable `transcript.html`.

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
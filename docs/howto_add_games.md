from clemcore.clemgame import ParseError

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
- `_does_game_proceed()` determines if the game play loop should continue, enforcing game rules
- `_parse_response(self.current_player, response)` processes the player's response, checking it for game rule conformity 
and extracting game-relevant response content to return
- `_advance_game(self.current_player, parsed_response)` uses the processed player response to update the game state
- `compute_turn_score()` calculates a score for the current turn, while mandatory, it is only used for reinforcement 
learning (PlayPen)
- `compute_episode_score()` calculates a score for the entire, while mandatory, it is only used for reinforcement 
learning (PlayPen), and is not called as part of the normal game play loop

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
over in the order they were added, and the next round starts when both players have responded.

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

Next we define how responses are checked for following the prompted format and how valid responses are processed.

```python
def _parse_response(player: Player, response) -> str:
    if player == self.guesser:
        # validate response format
        if not response.startswith("GUESS:"):
            self.invalid_response = True
            raise ParseError
        self.log_to_self("valid response", "continue")
        # extract guess word
        guess_word = response.replace("GUESS:", "")
        guess_word = guess_word.strip()
        guess_word = guess_word.lower()
        guess_word = string_utils.remove_punctuation(guess_word)
        self.guess_word = guess_word.lower()
        self.log_to_self("parsed guess", self.guess_word)
        return guess_word
    if player == self.describer:
        # validate response format
        if not response.startswith("CLUE:"):
            self.invalid_response = True
            raise ParseError
        self.log_to_self("valid response", "continue")
        clue = response.replace("CLUE:", "")
        clue = clue.strip()
        clue = clue.lower()
        clue = string_utils.remove_punctuation(clue)
        self.log_to_self("parsed clue", clue)
        return clue
```

We implement a function outside the DialogueGameMaster subclass for checking if a clue is following the game rules to 
keep the code concise.

```python
def check_clue(clue: str, target_word: str, related_words: List[str],
               stemmer=EN_STEMMER, return_clue=False) -> Union[Tuple[str, List[Dict]], List[Dict]]:
    clue_words = clue.split(" ")
    clue_words = [clue_word for clue_word in clue_words if clue_word not in EN_STOPWORDS]
    clue_word_stems = [stemmer.stem(clue_word) for clue_word in clue_words]
    errors = []
    target_word_stem = stemmer.stem(target_word)
    related_word_stems = [stemmer.stem(related_word) for related_word in related_words]

    for clue_word, clue_word_stem in zip(clue_words, clue_word_stems):
        if target_word_stem == clue_word_stem:
            errors.append({
                "message": f"Target word '{target_word}' (stem={target_word_stem}) "
                           f"is similar to clue word '{clue_word}' (stem={clue_word_stem})",
                "type": 0
            })
        for related_word, related_word_stem in zip(related_words, related_word_stems):
            if related_word_stem == clue_word_stem:
                errors.append({
                    "message": f"Related word '{related_word}' (stem={related_word_stem}) "
                               f"is similar to clue word '{clue_word}' (stem={clue_word_stem})",
                    "type": 1
                })
    if return_clue:
        return clue, errors
    return errors
```

After the responses are checked for correct format and parsed, they are passed to the method checking the game rules 
proper, which we implement in the DialogueGameMaster subclass. 

```python
def _advance_game(self, player: Player, parsed_response: str):
    if player == self.describer:
        # validate clue
        clue, errors = check_clue(parsed_response, self.target_word, self.related_words, return_clue=True)
        if errors:
            error = errors[0]  # highlight single error
            self.clue_error = error
            raise GameError
        self.log_to_self("valid clue", clue)
        # pass valid clue to guesser
        self.set_context_for(self.guesser, f"CLUE: {clue}")
    if player == self.guesser:
        # pass guess to clue giver
        self.set_context_for(self.describer, f"GUESS: {parsed_response}")
```

We calculate turn and episode scores next.

```python
def compute_turn_score(self):
    return 1 if self.is_success() else 0

def compute_episode_score(self):
    if self.is_success():
        return 100 / (self.current_round + 1)  # zero-based
    return 0
```

Then we must decide if the word describing and guessing should continue.

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

def is_terminal(self):
    if self.is_aborted():
        return True
    if self.is_failure():
        return True
    if self.is_success():
        return True
    return False

def is_aborted(self):
    return self.invalid_response

def is_failure(self):
    if self.is_clue_error():
        return True
    if self.is_turn_limit_reached():
        return True
    return False

def is_clue_error(self):
    return self.clue_error is not None

def is_turn_limit_reached(self):
    return self.current_round >= self.max_rounds

def is_success(self):
    return self.guess_word == self.target_word
```

Having implemented these methods and function, we have a functioning DialogueGameMaster subclass for the Taboo game.

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

For example, the taboo game guesser agent can be implemented as a player that can be a cLLM or replies with a static 
response that always guesses the word "pear":

```python
from clemcore.clemgame import Player

class WordGuesser(Player):

   def __init__(self, model_name):
      super().__init__(model_name)

   def _custom_response(self, messages, turn_idx):
      # mock response
      return f'Pear'
```

### GameScorer class
The GameScorer class takes episode records and calculates round and episode scores.  
The GameScorer subclass for the Taboo game does so with a single method overriding the base class's core scoring method:
```python
class TabooScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        if "meta" in episode_interactions:  # if given, copy over meta info
            self.scores["meta"] = episode_interactions["meta"]
        if "player_models" in episode_interactions:  # if given, copy over players info
            self.scores["player_models"] = episode_interactions["player_models"]
        if "players" in episode_interactions:  # if given, copy over players info
            self.scores["players"] = episode_interactions["players"]
        """ Episode level scores"""
        turn_scores = []
        prev_guess = None
        prev_guess_counter = 0
        prev_clue = None
        prev_clue_counter = 0
        invalid_response = False  # Note: This only takes into consideration that both players were compliant or not
        guesser_won = False
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            turn_score = {"guess": None, "clue": None, "request_count": 1}

            for event in turn:
                action = event["action"]
                if action["type"] == "invalid format":
                    invalid_response = True
                if action["type"] == "guess":
                    turn_score["guess"] = action["content"]
                if action["type"] == "clue":
                    turn_score["clue"] = action["content"]
                if action["type"] == "correct guess":
                    guesser_won = True

            if invalid_response:
                turn_score["violated_request_count"] = 1
                turn_score["parsed_request_count"] = 0
            else:
                turn_score["violated_request_count"] = 0
                turn_score["parsed_request_count"] = 1

            if turn_score["guess"] is not None and turn_score["guess"] == prev_guess:  # might be None, if clue is wrong
                prev_guess_counter += 1
            if turn_score["clue"] is not None and turn_score["clue"] == prev_clue:
                prev_clue_counter += 1
            self.log_turn_score(turn_idx, 'Accuracy', 1 if guesser_won else 0)
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_VIOLATED, turn_score["violated_request_count"])
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_PARSED, turn_score["parsed_request_count"])
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT, turn_score["request_count"])
            prev_guess = turn_score["guess"]
            prev_clue = turn_score["clue"]
            turn_scores.append(turn_score)

        violated_request_count = sum([turn["violated_request_count"] for turn in turn_scores])
        self.log_episode_score(METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)

        parsed_request_count = sum([turn["parsed_request_count"] for turn in turn_scores])
        self.log_episode_score(METRIC_REQUEST_COUNT_PARSED, parsed_request_count)

        request_count = sum([turn["request_count"] for turn in turn_scores])
        self.log_episode_score(METRIC_REQUEST_COUNT, request_count)

        self.log_episode_score(METRIC_REQUEST_SUCCESS_RATIO, parsed_request_count / request_count)
        # checking the last guess (could be None) is ok,
        # b.c. the game ends only successfully, when there is a correct guess

        # Common metrics
        if invalid_response:  # whether a violation of the game rules happened (response not parsable)
            self.log_episode_score(METRIC_ABORTED, 1)
            self.log_episode_score(METRIC_SUCCESS, 0)
            self.log_episode_score(METRIC_LOSE, 0)
            # Game-specific metrics
            self.log_episode_score(BENCH_SCORE, np.nan)  # metric not applicable
        else:
            self.log_episode_score(METRIC_ABORTED, 0)
            if guesser_won:
                self.log_episode_score(METRIC_SUCCESS, 1)
                self.log_episode_score(METRIC_LOSE, 0)
                self.log_episode_score(BENCH_SCORE, 100 / len(turn_scores))  # how early the guesser found the word
            else:
                self.log_episode_score(METRIC_SUCCESS, 0)
                self.log_episode_score(METRIC_LOSE, 1)
                self.log_episode_score(BENCH_SCORE, 0)  # word not found

        # Game-specific metrics
        # How often the Guesser repeated a guess
        self.log_episode_score('Repetition-Guesser', prev_guess_counter)
        # How often the Describer repeated itself
        self.log_episode_score('Repetition-Describer', prev_clue_counter)
        # this might require a side-loop between describer and GM (game should not continue with Guesser)
        # self.log_episode_score('Rule-following', ...)
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
from clemcore.clemgame import GameInstanceGenerator

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
Add to the module a `master.py` that implements the `GameMaster` and `GameScorer`, and a `clemgame.json`.  

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
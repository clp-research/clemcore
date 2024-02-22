from typing import Dict, List, Tuple, Set
from string import Template
import random, string, re, statistics, math, nltk

from clemgame.clemgame import GameMaster, GameScorer, GameBenchmark, Player, DialogueGameMaster
from clemgame import get_logger
from clemgame.metrics import METRIC_ABORTED, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, BENCH_SCORE
from games.codenames.constants import *
from games.codenames.validation_errors import *

logger = get_logger(__name__)

nltk.download('wordnet', quiet=True)
EN_LEMMATIZER = nltk.stem.WordNetLemmatizer()

# TODO: reuse players for other codename variants, e.g. Duet?
# TODO: check whether target is only a number -> validation error, not ignoring targets!

def find_line_starting_with(prefix, lines):
    for line in lines:
        if line.startswith(prefix):
            return line

class Guesser(Player):
    def __init__(self, model_name: str, flags: Dict[str, bool]):
        super().__init__(model_name)
        self.guesses: List[str] = ['guess', 'word']
        self.prefix: str = "GUESS: "
        self.retries: int = 0
        self.flags = flags
        self.flags_engaged = {key: 0 for key, value in flags.items()}

    def _custom_response(self, history, turn) -> str:
        prompt = history[-1]["content"]
        board = prompt.split('\n\n')[1].split(', ')
        number_of_allowed_guesses = int(re.search(r"up to ([0-9]+) words", prompt).group(1))
        self.guesses = random.sample(board, number_of_allowed_guesses)
        self.guesses = [word.strip('. ') for word in self.guesses]
        return self.recover_utterance()
    
    def validate_response(self, utterance: str, remaining_words: List[str], number_of_allowed_guesses: int):
        # utterance should only contain one line
        if '\n' in utterance:
            if self.flags["IGNORE RAMBLING"]:
                line = find_line_starting_with(self.prefix, utterance.split('\n'))
                self.flags_engaged["IGNORE RAMBLING"] += 1
                if line:
                    utterance = line
            else:
                raise GuesserRamblingError(utterance)
        # utterance needs to start with GUESS
        if not utterance.startswith(self.prefix):
            raise MissingGuessPrefix(utterance, self.prefix)
        utterance = utterance.removeprefix(self.prefix)
        
        guesses = utterance.split(', ')
        for guess in guesses:
            if any(character in guess for character in CHARS_TO_STRIP):
                if self.flags["STRIP WORDS"]:
                    self.flags_engaged["STRIP WORDS"] += 1
                else:
                    raise GuessContainsInvalidCharacters(utterance, guess)
        if self.flags["STRIP WORDS"]:
            guesses = [word.strip(CHARS_TO_STRIP) for word in guesses]
        guesses = [guess.lower() for guess in guesses]
        # must contain one valid guess, but can only contain $number guesses max
        if not (0 < len(guesses) <= number_of_allowed_guesses):
            raise WrongNumberOfGuessesError(utterance, guesses, number_of_allowed_guesses)
        # guesses must be words on the board that are not revealed yet
        for guess in guesses:
            if not guess in remaining_words:
                if self.flags["IGNORE FALSE TARGETS OR GUESSES"]:
                    self.flags_engaged["IGNORE FALSE TARGETS OR GUESSES"] += 1
                else:
                    raise InvalidGuessError(utterance, guess, remaining_words)
            
    def parse_response(self, utterance: str) -> str:
        if self.flags["IGNORE RAMBLING"]:
            utterance = find_line_starting_with(self.prefix, utterance.split('\n'))
        utterance = utterance.removeprefix(self.prefix)
        self.guesses = utterance.split(', ')
        self.guesses = [word.strip(CHARS_TO_STRIP).lower() for word in self.guesses]
        return f"{', '.join(self.guesses)}"
            
    def recover_utterance(self) -> str:
        return f"{self.prefix}{', '.join(self.guesses)}"

class ClueGiver(Player):
    def __init__(self, model_name: str, flags: Dict[str, bool]):
        super().__init__(model_name)
        self.clue_prefix: str = "CLUE: "
        self.target_prefix: str = "TARGETS: "
        self.clue: str = 'clue'
        self.number_of_targets: int = 2
        self.targets: List[str] = ['target', 'word']
        self.retries: int = 0
        self.flags = flags
        self.flags_engaged = {key: 0 for key, value in flags.items()}

    def _custom_response(self, history, turn) -> str:
        prompt = history[-1]["content"]
        match = re.search(r"team words are: (.*)\.", prompt)
        if match != None:
            # Player was actually prompted (otherwise it was reprompted and the team_words stay the same)
            team_words = match.group(1)
            team_words = team_words.split(', ')
            self.targets = random.sample(team_words, min(2, len(team_words)))
        self.number_of_targets = len(self.targets)
        self.clue = "".join(random.sample(list(string.ascii_lowercase), 6))
        return self.recover_utterance(with_targets=True)

    def check_morphological_similarity(self, utterance, clue, board):
        # lemma checks
        clue_lemma = EN_LEMMATIZER.lemmatize(clue)
        board_words_lemmas = [EN_LEMMATIZER.lemmatize(word) for word in board]
        if clue_lemma in board_words_lemmas:
            similar_board_word = board[board_words_lemmas.index(clue_lemma)]
            raise RelatedClueError(utterance, clue, similar_board_word)
    
    def validate_response(self, utterance: str, remaining_words: List[str]):
        # utterance should contain two lines, one with the clue, one with the targets
        parts = utterance.split('\n')
        if len(parts) < 1:
            raise TooFewTextError(utterance)
        elif len(parts) > 2:
            if not self.flags["IGNORE RAMBLING"]:
                raise CluegiverRamblingError(utterance)
            else:
                self.flags_engaged["IGNORE RAMBLING"] += 1

        clue = find_line_starting_with(self.clue_prefix, parts)
        targets = find_line_starting_with(self.target_prefix, parts)
        if not clue:
            raise MissingCluePrefix(utterance, self.clue_prefix)
        if not targets:
            raise MissingTargetPrefix(utterance, self.target_prefix)
        
        clue = clue.removeprefix(self.clue_prefix).lower()
        if any(character in clue for character in CHARS_TO_STRIP):
            if self.flags["STRIP WORDS"]:
                self.flags_engaged["STRIP WORDS"] += 1
                clue = clue.strip(CHARS_TO_STRIP)
            else:
                raise ClueContainsNonAlphabeticalCharacters(utterance, clue)
        if any(character in clue for character in NUMBERS_TO_STRIP):
            if self.flags["IGNORE NUMBER OF TARGETS"]:
                self.flags_engaged["IGNORE NUMBER OF TARGETS"] += 1
                clue  = clue.strip(NUMBERS_TO_STRIP)
            else:
                raise ClueContainsNumberOfTargets(utterance, clue)

        targets = targets.removeprefix(self.target_prefix).split(', ')
        for target in targets:
            if any(character in target for character in CHARS_TO_STRIP):
                if self.flags["STRIP WORDS"]:
                    self.flags_engaged["STRIP WORDS"] += 1
        if self.flags["STRIP WORDS"]:
            targets = [target.strip(CHARS_TO_STRIP) for target in targets]
        targets = [target.lower() for target in targets]
        
        # Clue needs to be a single word
        if ' ' in clue:
            raise ClueContainsSpaces(utterance, clue)
        if not clue.isalpha():
            raise ClueContainsNonAlphabeticalCharacters(utterance, clue)
        # Clue needs to contain a word that is not morphologically similar to any word on the board
        # TODO: morphological relatedness!
        if clue in remaining_words:
            raise ClueOnBoardError(utterance, clue, remaining_words)
        
        for target in targets:
            if not target in remaining_words:
                if self.flags["IGNORE FALSE TARGETS OR GUESSES"]:
                    self.flags_engaged["IGNORE FALSE TARGETS OR GUESSES"] += 1
                else:
                    raise InvalidTargetError(utterance, target, remaining_words)
            
    def parse_response(self, utterance: str) -> str:
        parts = utterance.split('\n')
        clue = find_line_starting_with(self.clue_prefix, parts).removeprefix(self.clue_prefix)
        targets = find_line_starting_with(self.target_prefix, parts).removeprefix(self.target_prefix)
        self.clueclue = clue.lower().strip(CHARS_TO_STRIP).strip(NUMBERS_TO_STRIP)
        self.targets = targets.split(', ')
        self.targets = [target.strip(CHARS_TO_STRIP).lower() for target in self.targets]
        self.number_of_targets = len(self.targets)
        return f"{self.clue}, {self.number_of_targets}"

    def recover_utterance(self, with_targets = False) -> str:
        if with_targets:
            targets = ', '.join(self.targets)
            return f"{self.clue_prefix}{self.clue}\n{self.target_prefix}{targets}"
        return f"{self.clue_prefix}{self.clue}, {len(self.targets)}"

class CodenamesBoard:
    def __init__(self, team_words, opponent_words, innocent_words, assassin_words, flags):
        self.hidden = {TEAM: team_words, INNOCENT: innocent_words, OPPONENT: opponent_words, ASSASSIN: assassin_words}
        self.revealed = {TEAM: {TEAM: [], INNOCENT: [], OPPONENT: [], ASSASSIN: []},
                         OPPONENT: {TEAM: [], INNOCENT: [], OPPONENT: [], ASSASSIN: []}}
        self.flags = flags

    def get_current_board(self) -> Dict:
        return {HIDDEN: self.hidden, 
                REVEALED: self.revealed}

    def get_all_hidden_words(self) -> List:
        hidden_words = []
        for assignment in self.hidden:
            hidden_words.extend(self.hidden[assignment])
        return hidden_words

    def get_hidden_words(self, with_assignment: str) -> List:
        return self.hidden[with_assignment]

    def reveal_word(self, word: str, by: str = TEAM):
        for assignment in self.hidden:
            if word in self.hidden[assignment]:
                self.revealed[by][assignment].append(word)
                self.hidden[assignment].remove(word)
                return assignment

        if not self.flags["IGNORE FALSE TARGETS OR GUESSES"]:
            raise ValueError(f"Word '{word}' was not found amongst the hidden words on the board, cannot be revealed.")
    
    def should_continue_after_revealing(self, word: str, by: str = TEAM):
        return word in self.revealed[by][by]
    
    def has_team_won(self) -> bool:
        return len(self.hidden[TEAM]) == 0
    
    def has_team_won_through_assassin(self) -> bool:
        return len(self.revealed[OPPONENT][ASSASSIN]) >= 1

    def has_opponent_won(self) -> bool:
        return len(self.hidden[OPPONENT]) == 0

    def has_opponent_won_through_assassin(self) -> bool:
        return len(self.revealed[TEAM][ASSASSIN]) >= 1
    
class CodenamesGame(DialogueGameMaster):
    """This class implements a codenames game in which player A
    is giving a clue for a set of target words on a board, 
    which player B has to guess from the given clue.
    """

    def __init__(self, experiment: Dict, player_backends: List[str]):
        super().__init__(GAME_NAME, experiment, player_backends)
        # fetch experiment parameters
        self.experiment = experiment
        self.opponent_difficulty: bool = experiment[OPPONENT_DIFFICULTY]

        # save player interfaces
        self.model_a: str = player_backends[0]
        self.model_b: str = player_backends[1]
        
    def _on_setup(self, **game_instance):
        self.game_instance = game_instance  # fetch game parameters here
        self.board: CodenamesBoard = CodenamesBoard(game_instance[ASSIGNMENTS][TEAM], 
                                                    game_instance[ASSIGNMENTS][OPPONENT], 
                                                    game_instance[ASSIGNMENTS][INNOCENT],
                                                    game_instance[ASSIGNMENTS][ASSASSIN],
                                                    self.experiment["flags"])
        
        self.aborted: bool = False
        self.lost: bool = False
        self.assassin_won: bool = False
        self.invalid_response: bool = False
        self.number_of_turns = 0
        self.request_count = 0
        self.parsed_request_count = 0
        self.violated_request_count = 0

        # Create the players
        self.cluegiver: Player = ClueGiver(self.model_a, self.experiment["flags"])
        self.guesser: Player = Guesser(self.model_b, self.experiment["flags"])

        # Add the players: these will be logged to the records interactions.json
        # Note: During game play the players will be called in the order added here
        self.add_player(self.cluegiver)
        self.add_player(self.guesser)
    
    def _was_target(self, word: str):
        return word in self.cluegiver.targets

    def _get_cluegiver_prompt(self, initial = False) -> str:
        folder = "initial_prompts" if initial else "intermittent_prompts"
        path = f"resources/{folder}/prompt_cluegiver"
        prompt_cluegiver = self.load_template(path)

        team_words = ", ".join(self.board.get_hidden_words(TEAM))
        opponent_words = ", ".join(self.board.get_hidden_words(OPPONENT))
        innocent_words = ", ".join(self.board.get_hidden_words(INNOCENT))
        assassin_words = ", ".join(self.board.get_hidden_words(ASSASSIN))

        instance_prompt_cluegiver = Template(prompt_cluegiver).substitute(team_words= team_words, 
                                                                          opponent_words=opponent_words, 
                                                                          innocent_words=innocent_words, 
                                                                          assassin_words=assassin_words)
        return instance_prompt_cluegiver
    
    def _get_guesser_prompt(self, initial = False) -> str:
        folder = "initial_prompts" if initial else "intermittent_prompts"
        path = f"resources/{folder}/prompt_guesser"
        prompt_guesser = self.load_template(path)
        
        board = ", ".join(self.board.get_all_hidden_words())
        instance_prompt_guesser = Template(prompt_guesser).substitute(board=board, 
                                                                      clue=self.cluegiver.clue, 
                                                                      number=self.cluegiver.number_of_targets)
        return instance_prompt_guesser
    
    def _opponent_turn(self):
        # reveal as many opponent cards as the opponent difficulty
        opponent_words = random.sample(self.board.get_hidden_words(OPPONENT), self.opponent_difficulty)
        for word in opponent_words:
            assignment = self.board.reveal_word(word, OPPONENT)
            self.log_to_self(Turn_logs.OPPONENT_REVEALED, {"assignment": assignment, "word": word})
    
    def _on_before_game(self):
        pass
        # self.add_user_message(self.cluegiver, self._get_cluegiver_prompt(initial=True))

    def _on_before_turn(self, current_turn):
        # let mock opponent reveal their cards
        if self.number_of_turns > 0:
            self._opponent_turn()

        # add new cluegiver prompt
        self.cluegiver.retries = 0
        self.guesser.retries = 0
        self.number_of_turns += 1
        initial = True if self.number_of_turns == 1 else False
        self.add_user_message(self.cluegiver, self._get_cluegiver_prompt(initial))

    def _does_game_proceed(self) -> bool:
        # Determine if the game should proceed. This is also called once initially.
        continue_game = True
        if self.invalid_response:
            self.aborted = True
            continue_game = False
        
        # for the base version, a check is needed whether all team words from one team are revealed or the assassin is revealed
        if self.board.has_team_won():
            self.lost = False
            self.assassin_won = False
            continue_game = False
        elif self.board.has_opponent_won():
            self.lost = True
            self.assassin_won = False
            continue_game = False
        elif self.board.has_team_won_through_assassin():
            self.lost = False
            self.assassin_won = True
            continue_game = False
        elif self.board.has_opponent_won_through_assassin():
            self.lost = True
            self.assassin_won = True
            continue_game = False

        if not continue_game:
            self._log_game_end()
            return False
        return True

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        self.request_count += 1
        self.invalid_response = False
        if player == self.cluegiver:
            try:
                player.validate_response(utterance, self.board.get_all_hidden_words())
            except ValidationError as error:
                self.log_to_self(Turn_logs.VALIDATION_ERROR, error.get_dict())
                self.invalid_response = True
                self.violated_request_count += 1
                self.last_error_message = error.message
                # add response to history nonetheless... but without parsing it
                self.add_assistant_message(player, utterance)
        else:
            try:
                player.validate_response(utterance, self.board.get_all_hidden_words(), self.cluegiver.number_of_targets)
            except ValidationError as error:
                self.log_to_self(Turn_logs.VALIDATION_ERROR, error.get_dict())
                self.invalid_response = True
                self.violated_request_count += 1
                self.last_error_message = error.message
                # add response to history nonetheless... but without parsing it
                self.add_assistant_message(player, utterance)
        
        return not self.invalid_response
    
    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        self.parsed_request_count += 1
        if player == self.cluegiver:
            utterance = player.parse_response(utterance)
            self.log_to_self(Turn_logs.CLUE, player.clue)
            # TODO: log target assignments here!
            self.log_to_self(Turn_logs.TARGETS, player.targets)
            return utterance, False
        else:
            parsed_utterance = player.parse_response(utterance)
            self.log_to_self(Turn_logs.GUESSES, player.guesses)
            for guess in player.guesses:
                assignment = self.board.reveal_word(guess)
                if not assignment:
                    continue
                self.log_to_self(Turn_logs.TEAM_REVEALED, {"assignment": assignment, "word": guess})
                if self._was_target(guess):
                    self.log_to_self(Turn_logs.TARGET_REVEALED, {"assignment": assignment, "word": guess})
                if not self.board.should_continue_after_revealing(guess):
                    self.log_to_self("turn end after", guess)
                    break
                
            return parsed_utterance, False
        
    def _on_before_reprompt(self, player: Player):
        logger.debug("Reprompting...")
        player.retries += 1
        player.flags_engaged["REPROMPT ON ERROR"] += 1
        self.add_user_message(player, f"Your answer did not follow the requested format: {self.last_error_message}")
    
    def _should_reprompt(self, player: Player):
        # return False
        if player.flags["REPROMPT ON ERROR"]:
            if player.retries < MAX_RETRIES:
                return self.invalid_response
        return False
    
    def _after_add_player_response(self, player: Player, utterance: str):
        if player == self.cluegiver:
            initial = True if self.number_of_turns == 1 else False
            self.add_user_message(self.guesser, self._get_guesser_prompt(initial))

        else:
            self.add_user_message(self.cluegiver, utterance)

    def _log_game_end(self):
        # log everything that is needed for score calculation and game evaluation
        self.log_key(BOARD_STATUS, self.board.get_current_board())
        self.log_key(NUMBER_OF_TURNS, self.number_of_turns)
        self.log_key(METRIC_ABORTED, self.aborted)
        self.log_key(METRIC_LOSE, self.lost)
        self.log_key(GAME_ENDED_THROUGH_ASSASSIN, self.assassin_won)
        # METRIC_SUCCESS does not need to be logged as it is inferred from ABORTED and LOSE
        self.log_key(METRIC_REQUEST_COUNT, self.request_count)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.parsed_request_count)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_count)
        self.log_key("Cluegiver engaged flags", self.cluegiver.flags_engaged)
        self.log_key("Guesser engaged flags", self.guesser.flags_engaged)

class CodenamesScorer(GameScorer):
    def __init__(self, experiment_config, game_instance):
        super().__init__(GAME_NAME, experiment_config, game_instance)

    def log_turn_score(self, turn_idx, name, value, scale=False):
        if type(value) == int or type(value) == float:
            value = round(value, 3)
            value = value * 100 if scale else value
        super().log_turn_score(turn_idx, name, value)

    def log_episode_score(self, name, value, scale=False):
        value = round(value, 3)
        value = value * 100 if scale else value
        super().log_episode_score(name, value)

    def score_turns(self, episode_interactions):
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            # Metrics per turn:
            # target-precision, target-recall, target-f1
            # team-precision
            # invalid formats per player
            turn_score = {Turn_logs.TARGETS: [], CLUEGIVER: {Turn_logs.VALIDATION_ERROR: 0}, GUESSER: {Turn_logs.VALIDATION_ERROR: 0}, 
                          REVEALED: {TARGET: 0, TEAM: 0, OPPONENT: 0, INNOCENT: 0, ASSASSIN: 0, "total": 0}}
            for event in turn:
                action = event["action"]
                match action["type"]:
                    case Turn_logs.VALIDATION_ERROR:
                        player = action["content"]["player"]
                        turn_score[player][Turn_logs.VALIDATION_ERROR] += 1
                    case Turn_logs.TARGETS:
                        turn_score[Turn_logs.TARGETS] = action["content"]
                    case Turn_logs.TEAM_REVEALED:
                        turn_score[REVEALED][action["content"]["assignment"]] += 1
                        turn_score[REVEALED]["total"] += 1
                    case Turn_logs.TARGET_REVEALED:
                        turn_score[REVEALED][TARGET] += 1
            
            # TODO: calculate cluegiver target precision, more metrics concerning cluegiver and guesser precision, recall and f1
            # to calculate cluegiver target precision, I would need the board assignments that I do not have
            #cluegiver_target_precision = 0
            #for target in turn_score[Turn_logs.TARGETS]:
            #    if target in 
            #turn_score[]

            sum_revealed_words = turn_score[REVEALED][TEAM] + turn_score[REVEALED][OPPONENT] + turn_score[REVEALED][INNOCENT] + turn_score[REVEALED][ASSASSIN]
            target_precision = 0
            target_recall = 0
            target_f1 = 0
            team_precision = 0
            if sum_revealed_words:
                target_precision = turn_score[REVEALED][TARGET] / sum_revealed_words
                team_precision = turn_score[REVEALED][TEAM] / sum_revealed_words
            
            if len(turn_score[Turn_logs.TARGETS]) > 0:
                target_recall = turn_score[REVEALED][TARGET] / len(turn_score[Turn_logs.TARGETS])
            if target_precision + target_recall > 0:
                target_f1 = 2 * target_precision * target_recall / (target_precision + target_recall)
            
            self.log_turn_score(turn_idx, "turn", turn_score)
            self.log_turn_score(turn_idx, f"{CLUEGIVER} {Turn_logs.VALIDATION_ERROR}", turn_score[CLUEGIVER][Turn_logs.VALIDATION_ERROR])
            self.log_turn_score(turn_idx, f"{GUESSER} {Turn_logs.VALIDATION_ERROR}", turn_score[GUESSER][Turn_logs.VALIDATION_ERROR])
            self.log_turn_score(turn_idx, "target precision", target_precision)
            self.log_turn_score(turn_idx, "target recall", target_recall)
            self.log_turn_score(turn_idx, "target f1", target_f1)
            self.log_turn_score(turn_idx, "team precision", team_precision)

    def score_game(self, episode_interactions):
        # game-specific scores

        for flag_name, value in self.experiment["flags"].items():
            if value:
                self.log_episode_score(f"Cluegiver {flag_name}", episode_interactions["Cluegiver engaged flags"][flag_name])
                self.log_episode_score(f"Guesser {flag_name}", episode_interactions["Guesser engaged flags"][flag_name])       

        number_of_turns = episode_interactions[NUMBER_OF_TURNS]
        self.log_episode_score(NUMBER_OF_TURNS, number_of_turns)
        number_of_team_words = self.experiment["assignments"]["team"]
        efficiency_multiplier = 2 # expecting two team words to be revealed each turn
        efficiency = min(1/efficiency_multiplier * number_of_team_words * 1/number_of_turns, 1)
        self.log_episode_score("efficiency", efficiency)
        target_f1s = [self.scores["turn scores"][x]["target f1"] for x in self.scores["turn scores"]]
        avg_target_f1s = statistics.mean(target_f1s)
        self.log_episode_score("avg target f1", avg_target_f1s)

        # plus all required game scores
        super().score_game(episode_interactions)
    
    def score_game_end(self, episode_interactions):
        super().score_game_end(episode_interactions)
        # plus game specific things
        # won or lost through assassin or through revealing all words of one team

        # TODO: should ratios also be 0 or NaN when the game was aborted? yes they should...

        self.log_episode_score(GAME_ENDED_THROUGH_ASSASSIN, episode_interactions[GAME_ENDED_THROUGH_ASSASSIN])

        # self.board_at_end = episode_interactions[BOARD_STATUS]
        # self.log_episode_score(BOARD_STATUS, board_at_end)

        number_of_team_words = self.experiment["assignment"]["team"]
        number_of_non_team_words = self.experiment["assignment"]["opponent"] + self.experiment["assignment"]["innocent"] + self.experiment["assignment"]["assassin"]
        self.log_episode_score("episode recall", len(self.board_at_end[REVEALED][TEAM][TEAM]) / number_of_team_words)
        self.log_episode_score("episode negative recall", 1 - (len(self.board_at_end[REVEALED][TEAM][ASSASSIN]) + len(self.board_at_end[REVEALED][TEAM][OPPONENT]) + len(self.board_at_end[REVEALED][TEAM][INNOCENT])) / number_of_non_team_words)
       
    def log_main_score(self, episode_interactions):
        # all logged scores are available via self.scores["episode scores"][score_name]
        # or self.scores["turn scores"][turn_idx][score_name]

        if self.scores["episode scores"][METRIC_ABORTED]:
            self.log_episode_score(BENCH_SCORE, math.nan)
            return

        # Main Score: harmonic mean of success (revealed team words / all team words (recall)) and efficiency (1/number of turns)
        success = self.scores["episode scores"]["episode recall"]
        efficiency = self.scores["episode scores"]["efficiency"]
        main_score = statistics.harmonic_mean([success, efficiency])
        self.log_episode_score(BENCH_SCORE, main_score, scale=True)

class CodenamesGameBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)
        random.seed(SEED)

    def get_description(self) -> str:
        return "Codenames game between a cluegiver and a guesser"

    def create_game_master(self, experiment: Dict, player_backends: List[str]) -> GameMaster:
        return CodenamesGame(experiment, player_backends)

    def create_game_scorer(self, experiment_config, game_instance) -> GameScorer:
        return CodenamesScorer(experiment_config, game_instance)

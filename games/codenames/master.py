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

# TODO: log all flags when used!!!
IGNORE_RAMBLING = True
IGNORE_FALSE_TARGETS_OR_GUESSES = True
REPROMPT_ON_ERROR = True
STRIP = True
IGNORE_NUMBER_OF_TARGETS = True


nltk.download('wordnet', quiet=True)
EN_LEMMATIZER = nltk.stem.WordNetLemmatizer()


# TODO: reuse players for other codename variants, e.g. Duet?

def find_line_starting_with(prefix, lines):
    for line in lines:
        if line.startswith(prefix):
            return line

class Guesser(Player):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.guesses: List[str] = ['guess', 'word']
        self.prefix: str = "GUESS: "
        self.retries: int = 0
        self.ignored_guesses = 0
        self.ignored_rambles = 0

    def _custom_response(self, history, turn) -> str:
        prompt = history[-1]["content"]
        board = prompt.split('\n\n')[1].split(', ')
        number_of_allowed_guesses = int(re.search(r"up to ([0-9]+) words", prompt).group(1))
        self.guesses = random.sample(board, number_of_allowed_guesses)
        self.guesses = [word.strip('. <>\"') for word in self.guesses]       # was not an issue but also does not hurt
        return self.recover_utterance()
    
    def validate_response(self, utterance: str, remaining_words: List[str], number_of_allowed_guesses: int):
        # utterance should only contain one line
        if '\n' in utterance:
            if IGNORE_RAMBLING:
                line = find_line_starting_with(self.prefix, utterance.split('\n'))
                self.ignored_rambles += 1
                if line:
                    utterance = line
            else:
                raise GuesserRamblingError(utterance)
        # utterance needs to start with GUESS
        if not utterance.startswith(self.prefix):
            raise MissingGuessPrefix(utterance, self.prefix)
        utterance = utterance.removeprefix(self.prefix)
        
        guesses = utterance.split(', ')
        guesses = [word.strip('. <>\"').lower() for word in guesses]
        # must contain one valid guess, but can only contain $number guesses max
        if not (0 < len(guesses) <= number_of_allowed_guesses):
            raise WrongNumberOfGuessesError(utterance, guesses, number_of_allowed_guesses)
        # guesses must be words on the board that are not revealed yet
        for guess in guesses:
            if not guess in remaining_words:
                if IGNORE_FALSE_TARGETS_OR_GUESSES:
                    self.ignored_guesses += 1
                    # TODO: log this somehow
                else:
                    raise InvalidGuessError(utterance, guess, remaining_words)
            
    def parse_response(self, utterance: str) -> str:
        if IGNORE_RAMBLING:
            utterance = find_line_starting_with(self.prefix, utterance.split('\n'))
        utterance = utterance.removeprefix(self.prefix)
        self.guesses = utterance.split(', ')
        self.guesses = [word.strip('. <>\"').lower() for word in self.guesses]
        return f"{', '.join(self.guesses)}"
            
    def recover_utterance(self) -> str:
        return f"{self.prefix}{', '.join(self.guesses)}"

class ClueGiver(Player):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.clue_prefix: str = "CLUE: "
        self.target_prefix: str = "TARGETS: "
        self.clue: str = 'clue'
        self.number_of_targets: int = 2
        self.targets: List[str] = ['target', 'word']
        self.retries: int = 0
        self.ignored_targets = 0
        self.ignored_rambles = 0

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

    def check_morphological_similarity(self, clue, board):
        # lemma checks
        clue_lemma = EN_LEMMATIZER.lemmatize(clue_word)
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
            if not IGNORE_RAMBLING:
                raise CluegiverRamblingError(utterance)
            else:
                self.ignored_rambles += 1

        clue = find_line_starting_with(self.clue_prefix, parts)
        targets = find_line_starting_with(self.target_prefix, parts)
        if not clue:
            raise MissingCluePrefix(utterance, self.clue_prefix)
        if not targets:
            raise MissingTargetPrefix(utterance, self.target_prefix)
        

        clue = clue.removeprefix(self.clue_prefix).lower()
        if STRIP:
            clue = clue.strip('<>\"')
        if IGNORE_NUMBER_OF_TARGETS:
            clue  = clue.strip(', 1234567890')
        targets = targets.removeprefix(self.target_prefix).split(', ')
        targets = [target.strip(" .!'<>\"").lower() for target in targets]
        
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
                if IGNORE_FALSE_TARGETS_OR_GUESSES:
                    # self.log_to_self("IGNORE FALSE TARGET")
                    print('IGNORE FALSE TARGET')
                    self.ignored_targets += 1
                    # TODO: remove target from targets?
                else:
                    raise InvalidTargetError(utterance, target, remaining_words)
            
    def parse_response(self, utterance: str) -> str:
        parts = utterance.split('\n')
        clue = find_line_starting_with(self.clue_prefix, parts).removeprefix(self.clue_prefix)
        targets = find_line_starting_with(self.target_prefix, parts).removeprefix(self.target_prefix)
        clue = clue.lower()
        if STRIP:
            clue = clue.strip('<>\"')
        if IGNORE_NUMBER_OF_TARGETS:
            clue = clue.strip(', 1234567890')
        self.clue = clue
        self.targets = targets.split(', ')
        self.targets = [target.strip(' .<>\"').lower() for target in self.targets]
        self.number_of_targets = len(self.targets)
        return f"{self.clue}, {self.number_of_targets}"

    def recover_utterance(self, with_targets = False) -> str:
        if with_targets:
            targets = ', '.join(self.targets)
            return f"{self.clue_prefix}{self.clue}\n{self.target_prefix}{targets}"
        return f"{self.clue_prefix}{self.clue}, {len(self.targets)}"

class CodenamesBoard:
    def __init__(self, team_words, opponent_words, innocent_words, assassin_words):
        self.hidden = {TEAM: team_words, INNOCENT: innocent_words, OPPONENT: opponent_words, ASSASSIN: assassin_words}
        self.revealed = {TEAM: {TEAM: [], INNOCENT: [], OPPONENT: [], ASSASSIN: []},
                         OPPONENT: {TEAM: [], INNOCENT: [], OPPONENT: [], ASSASSIN: []}}

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

        if not IGNORE_FALSE_TARGETS_OR_GUESSES:
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
        self.experiment: str = experiment[NAME]
        self.opponent_difficulty: bool = experiment[OPPONENT_DIFFICULTY]

        # save player interfaces
        self.model_a: str = player_backends[0]
        self.model_b: str = player_backends[1]
        
    def _on_setup(self, **game_instance):
        self.game_instance = game_instance  # fetch game parameters here
        self.board: CodenamesBoard = CodenamesBoard(game_instance[ASSIGNMENTS][TEAM], 
                                                    game_instance[ASSIGNMENTS][OPPONENT], 
                                                    game_instance[ASSIGNMENTS][INNOCENT],
                                                    game_instance[ASSIGNMENTS][ASSASSIN])
        
        self.aborted: bool = False
        self.lost: bool = False
        self.invalid_response: bool = False
        self.number_of_turns = 0
        self.request_count = 0
        self.parsed_request_count = 0
        self.violated_request_count = 0

        # Create the players
        self.cluegiver: Player = ClueGiver(self.player_backends[0])
        self.guesser: Player = Guesser(self.player_backends[1])

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
            self.log_key(GAME_END, Game_ends.ABORTED)
            self.aborted = True
            continue_game = False
        
        # for the base version, a check is needed whether all team words from one team are revealed or the assassin is revealed
        if self.board.has_team_won():
            self.log_key(GAME_END, Game_ends.TEAM_WON)
            self.lost = False
            continue_game = False
        elif self.board.has_opponent_won():
            self.log_key(GAME_END, Game_ends.OPPONENT_WON)
            self.lost = True
            continue_game = False
        elif self.board.has_team_won_through_assassin():
            self.log_key(GAME_END, Game_ends.TEAM_WON_THROUGH_ASSASSIN)
            self.lost = False
            continue_game = False
        elif self.board.has_opponent_won_through_assassin():
            self.log_key(GAME_END, Game_ends.OPPONENT_WON_THROUGH_ASSASSIN)
            self.lost = True
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
            self.log_to_self(Turn_logs.TARGETS, self.cluegiver.targets)
            return utterance, False
        else:
            parsed_utterance = player.parse_response(utterance)
            for guess in player.guesses:
                assignment = self.board.reveal_word(guess)
                if not assignment:
                    continue
                self.log_to_self(Turn_logs.TEAM_REVEALED, {"assignment": assignment, "word": guess})
                if self._was_target(guess):
                    self.log_to_self(Turn_logs.TARGET_REVEALED, guess)
                if not self.board.should_continue_after_revealing(guess):
                    self.log_to_self("turn end after", guess)
                    break
                
            return parsed_utterance, False
        
    def _on_before_reprompt(self, player: Player):
        logger.debug("Reprompting...")
        player.retries += 1
        self.add_user_message(player, f"Your answer did not follow the requested format: {self.last_error_message}")
    
    def _should_reprompt(self, player: Player):
        # return False
        if REPROMPT_ON_ERROR:
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
        # METRIC_SUCCESS does not need to be logged as it is inferred from ABORTED and LOSE
        self.log_key(METRIC_REQUEST_COUNT, self.request_count)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.parsed_request_count)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_count)
        self.log_key("Cluegiver ignored rambling", self.cluegiver.ignored_rambles)
        self.log_key("Guesser ignored rambling", self.guesser.ignored_rambles)
        self.log_key("Cluegiver ignored false targets", self.cluegiver.ignored_targets)
        self.log_key("Guesser ignored false guesses", self.guesser.ignored_guesses) 

class CodenamesScorer(GameScorer):
    def __init__(self):
        super().__init__(GAME_NAME)

    def log_turn_score(self, turn_idx, name, value, scale=False):
        if type(value) == int or type(value) == float:
            value = round(value, 3)
            value = value * 100 if scale else value
        super().log_turn_score(turn_idx, name, value)

    def log_episode_score(self, name, value, scale=False):
        value = round(value, 3)
        value = value * 100 if scale else value
        super().log_episode_score(name, value)

    def score_turns(self):
        for turn_idx, turn in enumerate(self.episode_interactions["turns"]):
            # Metrics per turn:
            # target-precision, target-recall, target-f1
            # team-precision
            # invalid formats per player
            turn_score = {Turn_logs.TARGETS: [], Turn_logs.CLUEGIVER_INVALID_FORMAT: 0, Turn_logs.GUESSER_INVALID_FORMAT: 0, 
                          REVEALED: {TARGET: 0, TEAM: 0, OPPONENT: 0, INNOCENT: 0, ASSASSIN: 0, "total": 0}}
            for event in turn:
                action = event["action"]
                match action["type"]:
                    case Turn_logs.CLUEGIVER_INVALID_FORMAT:
                        turn_score[Turn_logs.CLUEGIVER_INVALID_FORMAT] += 1
                    case Turn_logs.GUESSER_INVALID_FORMAT:
                        turn_score[Turn_logs.GUESSER_INVALID_FORMAT] += 1
                    case Turn_logs.VALIDATION_ERROR:
                        if action["content"]["player"] == GUESSER:
                            turn_score[Turn_logs.GUESSER_INVALID_FORMAT] += 1
                        else:
                            turn_score[Turn_logs.CLUEGIVER_INVALID_FORMAT] += 1
                    case Turn_logs.TARGETS:
                        turn_score[Turn_logs.TARGETS] = action["content"]
                    case Turn_logs.TEAM_REVEALED:
                        turn_score[REVEALED][action["content"]] += 1
                        turn_score[REVEALED]["total"] += 1
                    case Turn_logs.TARGET_REVEALED:
                        turn_score[REVEALED][TARGET] += 1
            
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
            self.log_turn_score(turn_idx, Turn_logs.CLUEGIVER_INVALID_FORMAT, turn_score[Turn_logs.CLUEGIVER_INVALID_FORMAT])
            self.log_turn_score(turn_idx, Turn_logs.GUESSER_INVALID_FORMAT, turn_score[Turn_logs.GUESSER_INVALID_FORMAT])
            self.log_turn_score(turn_idx, "target precision", target_precision)
            self.log_turn_score(turn_idx, "target recall", target_recall)
            self.log_turn_score(turn_idx, "target f1", target_f1)
            self.log_turn_score(turn_idx, "team precision", team_precision)

    def score_game(self):
        # game-specific scores
        # log flags and counted metrics for those
        # TODO: log all these metrics!
        
        if REPROMPT_ON_ERROR:
            pass
        if IGNORE_RAMBLING:
            self.log_episode_score("Cluegiver ignored rambling", self.episode_interactions["Cluegiver ignored rambling"])
            self.log_episode_score("Guesser ignored rambling", self.episode_interactions["Guesser ignored rambling"])
        if IGNORE_FALSE_TARGETS_OR_GUESSES:
            self.log_episode_score("Cluegiver ignored false targets", self.episode_interactions["Cluegiver ignored false targets"])
            self.log_episode_score("Guesser ignored false guesses", self.episode_interactions["Guesser ignored false guesses"])
        if STRIP:
            pass
        if IGNORE_NUMBER_OF_TARGETS:
            pass
            #self.log_episode_score()

        number_of_turns = self.episode_interactions[NUMBER_OF_TURNS]
        self.log_episode_score(NUMBER_OF_TURNS, number_of_turns)
        # TODO: better efficiency: average two targets per turn? Everything better gets ignored
        # is that fair, when you have 9 words? For the last one you cannot average 2...
        efficiency = min(4.5 / number_of_turns, 1) # TODO: 4.5 magic number: team assignment length / 2, replace!
        self.log_episode_score("efficiency", efficiency)
        target_f1s = [self.scores["turn scores"][x]["target f1"] for x in self.scores["turn scores"]]
        avg_target_f1s = statistics.mean(target_f1s)
        self.log_episode_score("avg target f1", avg_target_f1s)

        # plus all required game scores
        super().score_game()
    
    def score_game_end(self):
        super().score_game_end()
        # plus game specific things
        # won or lost through assassin or through revealing all words of one team

        # TODO: should ratios also be 0 or NaN when the game was aborted?

        game_end = self.episode_interactions[GAME_END]#_THROUGH_ASSASSIN]
        game_ended_through_assassin = False       # assume that game was aborted
        match game_ended_through_assassin:
            case Game_ends.TEAM_WON | Game_ends.OPPONENT_WON:
                game_ended_through_assassin = False
            case Game_ends.TEAM_WON_THROUGH_ASSASSIN | Game_ends.OPPONENT_WON_THROUGH_ASSASSIN:
                game_ended_through_assassin = True
        self.log_episode_score(GAME_END, game_ended_through_assassin)

        self.board_at_end = self.episode_interactions[BOARD_STATUS]
        # self.log_episode_score(BOARD_STATUS, board_at_end)

        # TODO: magic numbers!!!
        self.log_episode_score("team words revealed/all team words", len(self.board_at_end[REVEALED][TEAM][TEAM]) / 9)
        self.log_episode_score("other words not revealed/all other words", 1 - (len(self.board_at_end[REVEALED][TEAM][ASSASSIN]) + len(self.board_at_end[REVEALED][TEAM][OPPONENT]) + len(self.board_at_end[REVEALED][TEAM][INNOCENT])) / 16)
       
    def log_main_score(self):
        # all logged scores are available via self.scores["episode scores"][score_name]
        # or self.scores["turn scores"][turn_idx][score_name]

        if self.scores["episode scores"][METRIC_ABORTED]:
            self.log_episode_score(BENCH_SCORE, math.nan)
            return

        # Main Score: harmonic mean of success (revealed team words / all team words (recall)) and efficiency (1/number of turns)
        success = self.scores["episode scores"]["team words revealed/all team words"]
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

    def create_game_scorer(self) -> GameScorer:
        return CodenamesScorer()
